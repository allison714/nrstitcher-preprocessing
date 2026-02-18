import streamlit as st
import os
import sys
import pandas as pd
from typing import Dict
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import core
import importlib
try:
    importlib.reload(core)
except:
    pass

from core import (
    load_files, parse_filename, validate_dataset, infer_tiff_metadata,
    generate_stitch_settings, generate_slurm_script, generate_local_script, generate_manifest,
    DatasetManifest, ScanOrder, ChannelOrder, map_index, xy_to_tile_idx
)

st.set_page_config(page_title="Slurm Run Bundle Generator", layout="wide")

st.title("Slurm-Ready Run Bundle Generator")
st.markdown("""
Generates configuration and scripts for stitching large datasets on Misha cluster (pi2/NRStitcher).
""")

# --- Sidebar Inputs ---
st.sidebar.header("Dataset Configuration")

# Default path for convenience (user specific)
default_path = r"C:\\Users\\allis\\OneDrive\\Desktop\\260212"
data_path = st.sidebar.text_input(
    "Raw Data Directory", 
    value=default_path,
    help="Path to the folder containing your raw TIFF files.\n\n"
         "**Windows:** `C:\\Users\\YourName\\ImageFolder`\n\n"
         "**macOS:** `/Users/YourName/ImageFolder`\n\n"
         "**Linux:** `/home/yourname/ImageFolder`"
)
dataset_name = st.sidebar.text_input("Dataset Name (Output Folder)", value="my_dataset")
output_base_dir = st.sidebar.text_input("Output Location", value=os.path.join(os.path.dirname(data_path) if data_path else ".", "run_bundles"))

st.sidebar.subheader("Metadata")
prefix_filter = st.sidebar.text_input("Filename Prefix Filter", value="ss_single_", help="Only files starting with this will be included.")

# Detected metadata placeholders
files = []
validation_res = {}
width_px = 0
height_px = 0
bit_depth = 0

if data_path and os.path.exists(data_path):
    files = load_files(data_path, prefix_filter)
    validation_res = validate_dataset(files)
    
    if files:
        # Try to infer metadata from first file
        try:
            w, h, bd = infer_tiff_metadata(os.path.join(data_path, files[0]))
            width_px = w
            height_px = h
            bit_depth = bd
        except Exception as e:
            st.sidebar.warning(f"Could not infer metadata: {e}")

# Display Verification
st.header("1. Dataset Validation")
if not data_path:
    st.info("Enter a data directory to begin.")
elif not os.path.exists(data_path):
    st.error("Directory does not exist.")
else:
    if validation_res.get('valid'):
        st.success(f"Validation Passed: {validation_res['message']}")
    else:
        st.warning(f"Validation Issues: {validation_res.get('message')}")
        if validation_res.get('missing_indices'):
            st.error(f"Missing Indices Samples: {validation_res['missing_indices'][:10]}")

    st.write(f"**Total Files Found:** {len(files)}")
    if files:
        st.write(f"**Sample File:** `{files[0]}`")

# --- Parameters Form ---
st.header("2. Run Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Dimensions")
    n_tiles_x = st.number_input("Tiles X", min_value=1, value=5)
    n_tiles_y = st.number_input("Tiles Y", min_value=1, value=5)
    z_slices = st.number_input("Z Slices", min_value=1, value=200)
    n_channels = st.number_input("Channels", min_value=1, value=1)

with col2:
    st.subheader("Geometry")
    # Enforce integer percentage overlap
    overlap_x = st.number_input("Overlap X (%)", min_value=0, max_value=100, value=15, step=1)
    overlap_y = st.number_input("Overlap Y (%)", min_value=0, max_value=100, value=15, step=1)
    
    st.write("**Voxel Size (µm)**")
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        voxel_x = st.number_input("X", value=0.200, format="%.3f")
    with vc2:
        voxel_y = st.number_input("Y", value=0.200, format="%.3f")
    with vc3:
        voxel_z = st.number_input("Z", value=0.200, format="%.3f")
    
with col3:
    st.subheader("Image Specs (Auto-Detected)")
    img_w = st.number_input("Width (px)", value=width_px)
    img_h = st.number_input("Height (px)", value=height_px)
    img_bd = st.number_input("Bit Depth", value=bit_depth)

col4, col5 = st.columns(2)
with col4:
    scan_order = st.selectbox("Scan Order", [e.value for e in ScanOrder], index=0, 
        help="**Column Serpentine (pan-ASLM):** X slow, Y fast. Columns alternate up/down.\n\n"
             "**Row Serpentine:** Y slow, X fast. Rows alternate left/right.\n\n"
             "**Raster:** Simple row-by-row, left to right.")
with col5:
    channel_order = st.selectbox("Channel Order", [e.value for e in ChannelOrder])

# Channel Metadata
channel_meta = []
if n_channels > 0:
    st.caption("Optional: Enter channel details for reference.")
    # 4 channels per row max to fit 2 inputs
    cols = st.columns(min(n_channels, 4))
    for i in range(n_channels):
        with cols[i % 4]:
            st.markdown(f"**Channel {i}**")
            name = st.text_input("Name", key=f"name_{i}", placeholder="e.g. DAPI", label_visibility="collapsed")
            wl = st.text_input("Wavelength", key=f"wl_{i}", placeholder="e.g. 488nm", label_visibility="collapsed")
            channel_meta.append((name, wl))

# Validation of counts
expected_total = n_tiles_x * n_tiles_y * z_slices * n_channels
curr_total = len(files)
st.metric("Expected File Count", expected_total, delta=curr_total - expected_total, delta_color="inverse")

if curr_total != expected_total:
    st.error(f"Count Mismatch! Found {curr_total}, expected {expected_total}. Check parameters.")

# --- Execution Config ---
st.header("3. Execution Configuration")

execution_mode = st.radio(
    "Target Environment",
    ("Misha Cluster (Slurm)", "Local Workstation"),
    index=1
)

# Initialize session state for Slurm params if not set
if 'slurm_partition' not in st.session_state: st.session_state['slurm_partition'] = 'day'
if 'slurm_time' not in st.session_state: st.session_state['slurm_time'] = '04:00:00'
if 'slurm_cpus' not in st.session_state: st.session_state['slurm_cpus'] = 8
if 'slurm_mem' not in st.session_state: st.session_state['slurm_mem'] = '64G'

slurm_params = {}
conda_config = {}

st.subheader("Conda / pi2 Backend Configuration")
with st.expander("Configure Backend Paths", expanded=True):
    # Set defaults based on mode
    if execution_mode == "Misha Cluster (Slurm)":
        def_conda_sh = "~/miniconda3/etc/profile.d/conda.sh"
        def_entry = "" # "auto" by default (leave empty)
        def_env = "pi2_env"
    else:
        # Local defaults
        def_conda_sh = r"C:\Users\allis\anaconda3\condabin\conda.BAT" 
        def_entry = "" 
        def_env = "stitch_app"

    # Initialize from state if possible logic? No, simple inputs
    c1, c2 = st.columns(2)
    with c1:
        conda_sh = st.text_input("Conda Init Script (conda.sh)", value=def_conda_sh, help="Path to conda.sh to source.")
        env_name = st.text_input("Conda Environment Name", value=def_env)
    with c2:
        entrypoint = st.text_input("Entrypoint Override (Optional)", value=def_entry, help="Only set if auto-detection fails (e.g., full path to executable).")
        
        # Auto-detect local resource
        resource_path = os.path.join(os.path.dirname(__file__), 'resources', 'pi2')
        d_drive_path = r"D:\pi2-v4.5-win-no-opencl"
        
        if os.path.exists(resource_path) and os.listdir(resource_path):
            default_pi2 = resource_path
            st.success("✅ Found locally 'vendored' pi2 in `resources/pi2`.")
        elif os.path.exists(d_drive_path):
            default_pi2 = d_drive_path
            st.success(f"✅ Auto-detected pi2 on D: drive: `{d_drive_path}`")
        else:
            default_pi2 = ""
        
        pi2_local_path = st.text_input(
            "Path to 'pi2' Source/Binaries", 
            value=default_pi2,
            placeholder=r"C:\Users\allis\code\pi2", 
            help="Folder containing the 'pi2' python package (and .pyd files for Windows). Download from GitHub Releases."
        )
            
        # Auto detection helpers
        if st.button("Auto-detect (Local)"):
            import shutil
            
            # Check for conda
            conda_path = shutil.which("conda")
            if conda_path:
                st.success(f"Found 'conda' at: {conda_path}")
                
                # List environments
                try:
                    import subprocess
                    import json
                    # Use full path or just 'conda'
                    cmd = [conda_path, "env", "list", "--json"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        env_data = json.loads(result.stdout)
                        envs = [os.path.basename(p) for p in env_data.get('envs', [])]
                        st.info(f"Available Environments: {', '.join(envs)}")
                        st.caption("Copy one of these names into 'Conda Environment Name' if it contains pi2.")
                except Exception as e:
                    st.warning(f"Could not list environments: {e}")
            else:
                st.warning("'conda' command not found in PATH.")
                
            # Check for stitcher binaries
            found = shutil.which("nrstitcher") or shutil.which("pi2")
            if found:
                st.success(f"Found stitcher binary: {found}")
            else:
                if default_pi2:
                    st.info(f"✅ **Stitcher Status**: Ready (Found binary at `{default_pi2}`)")
                else:
                    try:
                        import pi2
                        st.info(f"✅ **Stitcher Status**: Installed in current env (`{pi2.__file__}`)")
                    except ImportError:
                        st.warning("⚠️ **Stitcher Status**: Not found in current env. (This is OK if you are generating a bundle for another machine or using the D: drive binary, but check the path above!)")
                    except:
                        pass

    conda_config = {
        'conda_sh': conda_sh,
        'env_name': env_name,
        'entrypoint': entrypoint
    }

if execution_mode == "Misha Cluster (Slurm)":
    st.subheader("Slurm Resources")
    
    # Auto-Recommend Controls
    col_auto, col_mode = st.columns(2)
    with col_auto:
        auto_recommend = st.checkbox("Auto-Recommend Resources?", value=False)
    with col_mode:
        rec_mode = st.radio("Run Mode", ["Production (day/week)", "Calibration (devel)"], index=0)
        
    if auto_recommend:
        # Default speed since benchmark removed
        default_speed = 100.0 
            
        # create temp manifest for estimation
        tmp_manifest = DatasetManifest(
            dataset_name=dataset_name, n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, z_slices=z_slices, n_channels=n_channels,
            overlap_x=int(overlap_x), overlap_y=int(overlap_y), voxel_size_x_um=voxel_x, voxel_size_y_um=voxel_y, voxel_size_z_um=voxel_z,
            scan_order=scan_order, channel_order=channel_order, width_px=img_w, height_px=img_h, bit_depth=img_bd,
            prefix_filter=prefix_filter, files=files
        )
        
        from core import estimate_resources
        mode_key = "production" if "Production" in rec_mode else "calibration"
        # Passing default speed
        est = estimate_resources(tmp_manifest, default_speed, mode=mode_key)
        
        # Update session state
        st.session_state['slurm_partition'] = est['partition']
        st.session_state['slurm_cpus'] = est['cpus']
        st.session_state['slurm_mem'] = est['mem']
        st.session_state['slurm_time'] = est['time']
        
        st.caption(f"Recommendation: {est['details']}")

    scol1, scol2 = st.columns(2)
    with scol1:
        partition = st.text_input("Partition", key='slurm_partition')
        time_limit = st.text_input("Time Limit", key='slurm_time')
    with scol2:
        cpus = st.number_input("CPUs per Task", min_value=1, key='slurm_cpus')
        mem = st.text_input("Memory", key='slurm_mem')
    
    slurm_params = {
        'partition': partition,
        'time': time_limit,
        'cpus': cpus,
        'mem': mem
    }
else:
    st.info("Local execution scripts (run_local.bat/.sh) will be generated.")

# Output Format Selection
st.subheader("Output Format")

# Override default red multiselect tag color to friendly teal
st.markdown("""
<style>
span[data-baseweb="tag"] {
    background-color: #0d9488 !important;
}
</style>
""", unsafe_allow_html=True)

output_formats = st.multiselect(
    "Select output format(s)",
    ["Raw (.raw)", "Zarr", "OME-TIFF (.ome.tif)"],
    default=["OME-TIFF (.ome.tif)"],
    help=(
        "**Raw (.raw):** Flat binary file — just voxel data, no headers. "
        "Fastest to write, but you'll need to know the dimensions (X×Y×Z) to open it. "
        "Useful if you plan to process the data further with custom scripts.\n\n"
        "**Zarr:** Chunked, multiscale format. Great for cloud storage and lazy loading "
        "of large volumes (e.g. with Napari or neuroglancer).\n\n"
        "**OME-TIFF (.ome.tif):** The gold standard for bioimage data. Embeds voxel sizes "
        "and metadata directly in the file. Opens natively in Fiji, Napari, QuPath, Imaris, etc."
    )
)

# --- Generation ---
st.header("4. Generate Bundle")

# Tile Preview
with st.expander("🔎 Preview Tiles (Verify Data)", expanded=False):
    if not files:
        st.info("No files loaded.")
    else:
        st.write(f"**Total files loaded:** `{len(files)}`")
        
        preview_tab1, preview_tab2 = st.tabs(["📷 Single Tile", "🔲 Grid View (up to 3×3)"])
        
        with preview_tab1:
            # Selector
            preview_idx = st.number_input(
                "Select File Number (Index)", 
                min_value=0, 
                max_value=len(files)-1, 
                value=0, 
                step=1,
                help="Choose which file to preview by its number in the list (0 is the first file)."
            )
            
            selected_file = files[preview_idx]
            full_path = os.path.join(data_path, selected_file)
            
            # Calc metadata
            t_idx, z_idx, c_idx = map_index(preview_idx, n_channels, z_slices)
            st.write(f"**Filename:** `{selected_file}`")
            
            # Format mapping string with optional metadata
            meta_str = ""
            if c_idx < len(channel_meta):
                name, wl = channel_meta[c_idx]
                parts = [p for p in [name, wl] if p and p.strip()]
                if parts:
                    meta_str = f" (**{' - '.join(parts)}**)"
            
            st.markdown(f"""
            **Mapping Indices:**
            *   **Tile (XY)**: `{t_idx}`
            *   **Z-Slice**: `{z_idx}`
            *   **Channel**: `{c_idx}` {meta_str}
            """)
            
            if os.path.exists(full_path):
                from core import get_tile_preview
                import importlib
                import core as _core_mod
                importlib.reload(_core_mod)
                from core import get_tile_preview
                
                img, err, stats = get_tile_preview(full_path)
                
                if img is not None:
                    st.image(img, caption=f"Preview (Auto B/C) - {selected_file}", use_container_width=True, clamp=True)
                    if stats:
                        st.caption(f"Stats: Min={stats['orig_min']:.1f}, Max={stats['orig_max']:.1f}, Type={stats['dtype']}")
                else:
                    st.error(f"Could not load image: {err}")
            else:
                st.error("File not found on disk.")
        
        with preview_tab2:
            # Grid View - up to 3x3 tiles
            grid_cols_count = min(3, n_tiles_x)
            grid_rows_count = min(3, n_tiles_y)
            
            st.write(f"Showing **{grid_cols_count}×{grid_rows_count}** tile grid (of {n_tiles_x}×{n_tiles_y} total)")
            
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                if z_slices > 1:
                    grid_z = st.slider("Z-Slice", min_value=0, max_value=z_slices-1, value=z_slices // 2, key="grid_z")
                else:
                    grid_z = 0
                    st.caption("Z-Slice: 0 (single slice)")
            with gcol2:
                if n_channels > 1:
                    grid_ch = st.slider("Channel", min_value=0, max_value=n_channels-1, value=0, key="grid_ch")
                else:
                    grid_ch = 0
                    st.caption("Channel: 0 (single channel)")
            
            # --- Mini-Map Visualization ---
            if n_tiles_x > 0 and n_tiles_y > 0:
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    
                    # Create detailed figure
                    # ... [existing matplotlib code] ...
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Draw the full grid (representing the dataset)
                    ax.set_xlim(-0.5, n_tiles_x - 0.5)
                    ax.set_ylim(-0.5, n_tiles_y - 0.5)
                    ax.set_aspect('equal')
                    ax.set_xlabel("Tile X")
                    ax.set_ylabel("Tile Y")
                    ax.set_title(f"Full Dataset Map ({n_tiles_x} x {n_tiles_y})")
                    
                    # Current Window (Red)
                    sel_w = min(3, n_tiles_x - grid_offset_x)
                    sel_h = min(3, n_tiles_y - grid_offset_y)
                    
                    rect = patches.Rectangle(
                        (grid_offset_x - 0.5, grid_offset_y - 0.5), 
                        sel_w, sel_h, 
                        linewidth=2, edgecolor='#e11d48', facecolor='#e11d48', alpha=0.3, label="Preview Window"
                    )
                    
                    # Draw grid dots
                    all_x = []
                    all_y = []
                    for y in range(n_tiles_y):
                        for x in range(n_tiles_x):
                            all_x.append(x)
                            all_y.append(y)
                    
                    ax.scatter(all_x, all_y, c='#94a3b8', marker='s', s=100, label="Tiles") 
                    ax.add_patch(rect)
                    
                    # Highlight active window tiles
                    act_x = []
                    act_y = []
                    for row_i in range(grid_rows_count):
                        for col_i in range(grid_cols_count):
                                gx = grid_offset_x + col_i
                                gy = grid_offset_y + row_i
                                act_x.append(gx)
                                act_y.append(gy)
                    ax.scatter(act_x, act_y, c='#e11d48', marker='s', s=100) 
                    
                    st.pyplot(fig, width="content")
                    st.caption("The **Red Box** above shows which part of the dataset is being previewed below.")
                    
                except ImportError:
                    st.warning("⚠️ Install `matplotlib` to see the dataset Mini-Map here.")
                    st.caption("Run: `pip install matplotlib` in your terminal.")

            # Offsets for panning across larger grids
            if n_tiles_x > grid_cols_count or n_tiles_y > grid_rows_count:
                ocol1, ocol2 = st.columns(2)
                with ocol1:
                    grid_offset_x = st.number_input("Start at Tile X", min_value=0, max_value=max(0, n_tiles_x - grid_cols_count), value=0, key="grid_ox")
                with ocol2:
                    grid_offset_y = st.number_input("Start at Tile Y", min_value=0, max_value=max(0, n_tiles_y - grid_rows_count), value=0, key="grid_oy")
            else:
                grid_offset_x = 0
                grid_offset_y = 0
            
            # Scan Order Preview Override
            all_orders = [e.value for e in ScanOrder]
            curr_order_idx = all_orders.index(scan_order) if scan_order in all_orders else 0
            
            grid_scan_order = st.selectbox(
                "Grid Scan Order (Preview Only)", 
                all_orders, 
                index=curr_order_idx,
                key="grid_order_select",
                help="Test different scan orders to verify alignment. Note: This only affects the preview; update Section 2 to apply to the actual run."
            )

            # Helper: tile_idx + z + ch -> linear file index
            def tile_to_file_idx(tile_idx, z, ch, n_ch, n_z):
                return tile_idx * (n_ch * n_z) + z * n_ch + ch
            
            # Lazy-load preview function
            from core import get_tile_preview
            import importlib
            import core as _core_mod
            importlib.reload(_core_mod)
            from core import get_tile_preview, xy_to_tile_idx # Ensure xy_to_tile_idx is imported
            
            # -- Render Composite Grid (Pixel Perfect) --
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                # We need the first valid image to know dimensions
                first_valid = None
                
                # Pre-scan for first valid image
                # Just check T0? Or iterate?
                # Let's assume standard size from first available.
                # Actually we can just load them on the fly.
                
                # To determine canvas size, we need W/H.
                # Let's try to load the very first tile in the window:
                # (grid_offset_x, (grid_offset_y + grid_rows_count - 1)) etc?
                # Simpler: just loop and load all into a dict first.
                
                grid_images = {} # Key: (row, col) -> Image
                tile_w, tile_h = 0, 0
                
                for row in range(grid_rows_count):
                    gy = (grid_offset_y + grid_rows_count - 1) - row
                    for col in range(grid_cols_count):
                        gx = grid_offset_x + col
                        
                        tile_idx = xy_to_tile_idx(gx, gy, n_tiles_x, n_tiles_y, grid_scan_order)
                        file_idx = tile_to_file_idx(tile_idx, grid_z, grid_ch, n_channels, z_slices)
                        
                        if file_idx < len(files):
                            fpath = os.path.join(data_path, files[file_idx])
                            if os.path.exists(fpath):
                                img_arr, _, _ = get_tile_preview(fpath) # Returns numpy array (RGB or Gray)
                                if img_arr is not None:
                                    # Convert numpy to PIL
                                    # Check limits
                                    pil_img = Image.fromarray(img_arr)
                                    grid_images[(row, col)] = (pil_img, tile_idx)
                                    if tile_w == 0:
                                        tile_w, tile_h = pil_img.size
                
                if tile_w > 0 and tile_h > 0:
                    # Calculate Canvas with Overlap
                    # Overlap is percentage of size? User inputs overlap_x (float) e.g. 10.0
                    ov_x_px = int(tile_w * (overlap_x / 100.0))
                    ov_y_px = int(tile_h * (overlap_y / 100.0))
                    
                    # Canvas Size
                    # Width = (W * Cols) - (Overlap * (Cols-1))
                    canvas_w = (tile_w * grid_cols_count) - (ov_x_px * (grid_cols_count - 1))
                    canvas_h = (tile_h * grid_rows_count) - (ov_y_px * (grid_rows_count - 1))
                    
                    # Ensure positive (overlap < 100%)
                    canvas_w = max(canvas_w, tile_w)
                    canvas_h = max(canvas_h, tile_h)
                    
                    composite = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
                    draw = ImageDraw.Draw(composite)
                    
                    for row in range(grid_rows_count):
                        for col in range(grid_cols_count):
                            if (row, col) in grid_images:
                                img, tidx = grid_images[(row, col)]
                                
                                # Position
                                # x = col * (W - Overlap)
                                pos_x = col * (tile_w - ov_x_px)
                                pos_y = row * (tile_h - ov_y_px)
                                
                                composite.paste(img, (pos_x, pos_y))
                                
                                # Draw Text
                                txt = f"T{tidx}"
                                # Default font
                                # Draw Top-Left with shadow for visibility
                                txt_pos = (pos_x + 5, pos_y + 5)
                                draw.text((txt_pos[0]+1, txt_pos[1]+1), txt, fill="black")
                                draw.text(txt_pos, txt, fill="white")
                                
                    st.image(composite, caption=f"Composite Preview ({overlap_x:.1f}% X / {overlap_y:.1f}% Y Overlap)", width="stretch")
                    st.caption("ℹ️ Labels are now positioned Top-Left on each tile.")
                else:
                    st.warning("No valid images found in this grid view region.")
                    
            except Exception as e:
                st.error(f"Error generating composite preview: {e}")
                # Fallback? No, just error.

            # Replaced Loop
            # for row in range(grid_rows_count):
            # ... [Old Loop Removed] ...


# Refactoring layout to put Tiles View toggle in Execution Config or right before Generate
st.subheader("Advanced Options")
symlink_help = """
**Why user might want this:**
1.  **Organization**: Renames your files into a clean `Tile_X_Y_Z.tif` format that the stitcher expects, without messing up your raw data.
2.  **Space Saving**: Symlinks are just shortcuts. It organizes the dataset without copying terabytes of images.
3.  **Verification**: You can look in the `tiles/` folder to verify the layout before running the heavy stitch.
"""
create_tiles_view = st.checkbox("Create Tiles View (Symlinks)?", value=False, help=symlink_help)
tiles_view_possible = True

# Hard Gating
if len(files) > 100000 or (n_tiles_x * n_tiles_y) > 2500:
    if create_tiles_view:
        st.warning("Tiles View force-disabled due to dataset size (>100k files or >2500 tiles).")
        create_tiles_view = False
        tiles_view_possible = False

generate_btn = st.button("Generate Run Bundle", disabled=(curr_total == 0))

if generate_btn:
    # Determine output folder suffix based on mode
    mode_suffix = "_slurm" if execution_mode == "Misha Cluster (Slurm)" else "_local"
    output_folder = f"{dataset_name}{mode_suffix}"
    output_dir = os.path.join(output_base_dir, output_folder)
    
    # Create manifest
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
        z_slices=z_slices,
        n_channels=n_channels,
        overlap_x=int(overlap_x),
        overlap_y=int(overlap_y),
        voxel_size_x_um=voxel_x,
        voxel_size_y_um=voxel_y,
        voxel_size_z_um=voxel_z,
        scan_order=scan_order,
        channel_order=channel_order,
        width_px=img_w,
        height_px=img_h,
        bit_depth=img_bd,
        prefix_filter=prefix_filter,
        files=files
    )
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        generate_manifest(manifest, output_dir)
        
        # Generate Tiles View FIRST if requested
        tiles_created_ok = False
        if create_tiles_view and tiles_view_possible:
            try:
                from core import generate_tiles_view
                count, errs = generate_tiles_view(manifest, output_dir, data_path)
                st.success(f"Generated {count} symlinks in 'tiles/' folder.")
                if errs:
                    st.warning(f"Encountered {len(errs)} errors (first few: {errs[:3]})")
                tiles_created_ok = True
            except Exception as e:
                st.error(f"Failed to create tiles view: {e}")
                st.info("On Windows, Symlinks require 'Developer Mode'. Falling back to absolute paths in config.")
                tiles_created_ok = False
        
        # Generate Stacking Script (Preprocessing)
        core.generate_stack_script(manifest, output_dir, data_path)

        # Determine the native stitch output format
        # nr_stitcher only supports 'raw' or 'zarr'
        want_ometiff = "OME-TIFF (.ome.tif)" in output_formats
        want_zarr = "Zarr" in output_formats
        want_raw = "Raw (.raw)" in output_formats
        
        # If user only wants OME-TIFF, we still need raw as intermediate
        stitch_fmt = "zarr" if (want_zarr and not want_raw and not want_ometiff) else "raw"
        
        # Generate Settings
        generate_stitch_settings(manifest, output_dir, data_path, use_tiles_view=tiles_created_ok, stitch_output_format=stitch_fmt)
        
        # Generate OME-TIFF converter if requested
        if want_ometiff:
            core.generate_ometiff_converter(manifest, output_dir)
        
        if execution_mode == "Misha Cluster (Slurm)":
            core.generate_slurm_script(manifest, slurm_params, output_dir, conda_config)
        else:
            embed_path = pi2_local_path if 'pi2_local_path' in locals() and pi2_local_path else None
            
            # Validation for Portable Bundle
            if embed_path and not os.path.exists(embed_path):
                st.error(f"Invalid 'pi2' source path: {embed_path}")
                st.stop()
                
            core.generate_local_script(manifest, output_dir, conda_config, embed_pi2_path=embed_path, convert_ometiff=want_ometiff)
            
            if not embed_path:
                st.warning("No 'pi2' source path provided. You MUST download 'pi2' manually and provide the path to create a portable bundle, or ensure it is installed in your environment.")
            else:
                st.success(f"Embedded 'pi2' from: {embed_path}")
        
        # Show output format summary
        fmt_list = []
        if want_raw: fmt_list.append("Raw (.raw)")
        if want_zarr: fmt_list.append("Zarr")
        if want_ometiff: fmt_list.append("OME-TIFF (.ome.tif)")
        st.info(f"📦 Output format(s): **{', '.join(fmt_list)}**")
        
        st.success(f"Successfully generated run bundle at: {output_dir}")
        st.balloons()
    except Exception as e:
        st.error(f"Error generating bundle: {e}")

# --- Author Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #555; font-size: 11px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;'>
        <strong title="I'm a developer now">A. Cairns</strong> || 2026<br>
        <span style="font-style: italic; color: #777;">Kuan x Bewersdorf Labs</span>
    </div>
    """, 
    unsafe_allow_html=True
)
