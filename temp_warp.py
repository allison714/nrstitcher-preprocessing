import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Inject source_tile extraction & Regional Edge Analysis & Plotly Interactive hover
replace_1 = r'''\1
                    source_tile = mags_data.get('source_tile', np.full(len(mags), "Unknown"))
                    max_tile = mags_data.get('max_tile_size', 300)'''

text = re.sub(r'(x_norm = mags_data\[\'x\'\]\n\s*y_norm = mags_data\[\'y\'\]\n)\s*max_tile = mags_data\.get\(\'max_tile_size\', 300\)', replace_1, text)


replace_2 = r'''                    if st.button("Generate Before/After Seam Script"):
                        seam_script_path = os.path.join(analysis_dir, "verify_overlap_seam.py")
                        seam_py = f"""#!/usr/bin/env python3
\"\"\"
Utility script to visualize the before/after difference of two overlapping tiles using the calculated pi2 shift fields.
Requires: pip install numpy scipy tifffile matplotlib
\"\"\"
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def load_warped_slice(tif_path, shift_raw_path, z_index, bin_scale={bin_scale}):
    print(f"Loading {{tif_path}} (Slice {{z_index}})...")
    with tifffile.TiffFile(tif_path) as tif:
        img_2d = tif.pages[z_index].asarray().astype(np.float32)
        
    print(f"Loading shift field {{shift_raw_path}}...")
    data = np.fromfile(shift_raw_path, dtype=np.float32)
    return img_2d

if __name__ == "__main__":
    print("To perform visual before/after seam verification, it is highly recommended to compare the")
    print("Rigid Stitch output (stitch_settings_rigid_preview.txt) vs the Non-Rigid stitched output in Fiji.")
"""
                        with open(seam_script_path, "w", encoding='utf-8') as f:
                            f.write(seam_py)
                        st.success(f"Generated `{seam_script_path}`!")
                        
                    st.markdown("##### Regional Edge Analysis")
                    st.markdown("Breaking down deformation by spatial quadrants helps diagnose directional drag or specific stage-axis slipping.")
                    
                    m_top = y_norm < 0.25
                    m_bot = y_norm > 0.75
                    m_left = x_norm < 0.25
                    m_right = x_norm > 0.75
                    
                    reg_data = []
                    for label, mask in [("Top Edge (y < 0.25)", m_top), ("Bottom Edge (y > 0.75)", m_bot), 
                                        ("Left Edge (x < 0.25)", m_left), ("Right Edge (x > 0.75)", m_right)]:
                        if np.any(mask):
                            reg_mags = mags[mask]
                            r_med = np.median(reg_mags)
                            r_max = np.max(reg_mags)
                            
                            row = {
                                "Region": label,
                                "Median Shift (px)": f"{r_med:.2f}",
                                "Worst Shift (px)": f"{r_max:.2f}"
                            }
                            if bin_scale > 1.0:
                                row["Median (px full-res)"] = f"{r_med * bin_scale:.2f}"
                            if voxel_x:
                                row["Median (µm)"] = f"{r_med * bin_scale * voxel_x:.2f}"
                            
                            reg_data.append(row)
                    if reg_data:
                        st.dataframe(pd.DataFrame(reg_data), hide_index=True, use_container_width=True)'''

# Target replacing the current 'if st.button("Generate Before/After Seam Script"):' block
text = re.sub(r'                    if st\.button\("Generate Before/After Seam Script"\):.*?(?=                # Sanity Check Panel)', replace_2 + '\n', text, flags=re.DOTALL)


# 2. Inject Defensibility subplots, replacing Plot 2 hexbin
replace_3 = r'''                with colB:
                    hexbin_help = """
The color of each hexagon corresponds to the median displacement magnitude of all the vectors that fall inside that specific spatial area:

*   **Dark Purple / Blue** regions indicate areas where the deformation was consistently very small, meaning the sample shape required very little non-linear stretching or shifting there.
*   **Light Green / Yellow** regions indicate "hotspots" where the alignment algorithm had to apply a much larger scale of local deformation to get the data to register properly across tiles.
"""
                    st.markdown("#### Hexbin Plot of Local Deformation Hotspots", help=hexbin_help)
                    plt.rcParams.update({'font.size': 8})
                    fig2 = plt.figure(figsize=(7.08, 3.5))
                    
                    # Panel A: Median Magnitude
                    ax2a = plt.subplot(1, 2, 1)
                    hb1 = ax2a.hexbin(x_norm, y_norm, C=mags, gridsize=30, cmap='viridis', 
                                   reduce_C_function=np.median, mincnt=1)
                    fig2.colorbar(hb1, ax=ax2a, label='Median Displ. (px)', fraction=0.046, pad=0.04)
                    ax2a.set_xlabel("Normalized X Coordinate")
                    ax2a.set_ylabel("Normalized Y Coordinate")
                    ax2a.set_xlim(0, 1)
                    ax2a.set_ylim(0, 1)
                    ax2a.invert_yaxis()
                    ax2a.set_aspect('equal', adjustable='box')
                    ax2a.grid(True, linestyle='--', alpha=0.3)
                    ax2a.set_title("Median Deformation Severity")
                    
                    # Panel B: Density Count (Defensibility)
                    ax2b = plt.subplot(1, 2, 2)
                    hb2 = ax2b.hexbin(x_norm, y_norm, gridsize=30, cmap='magma', mincnt=1)
                    fig2.colorbar(hb2, ax=ax2b, label='Vector Count (Density)', fraction=0.046, pad=0.04)
                    ax2b.set_xlabel("Normalized X Coordinate")
                    ax2b.set_xlim(0, 1)
                    ax2b.set_ylim(0, 1)
                    ax2b.invert_yaxis()
                    ax2b.set_aspect('equal', adjustable='box')
                    ax2b.grid(True, linestyle='--', alpha=0.3)
                    ax2b.set_title("Sample Density (Defensibility)")
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                st.markdown("#### Interactive Point-Cloud Drilldown", help="Hover over specific spatial regions to view exact sub-voxel deformation magnitudes. (Subsampled for web performance).")
                try:
                    import plotly.express as px
                    max_points = 15000
                    if len(mags) > max_points:
                        indices = np.random.choice(len(mags), max_points, replace=False)
                        px_x, px_y, px_mags = x_norm[indices], y_norm[indices], mags[indices]
                        px_dx, px_dy, px_dz = v_dx[indices], v_dy[indices], v_dz[indices]
                        px_src = source_tile[indices]
                    else:
                        px_x, px_y, px_mags = x_norm, y_norm, mags
                        px_dx, px_dy, px_dz = v_dx, v_dy, v_dz
                        px_src = source_tile
                        
                    plotly_df = pd.DataFrame({
                        'X (norm)': px_x, 'Y (norm)': px_y, 'Magnitude': px_mags,
                        'dx': px_dx, 'dy': px_dy, 'dz': px_dz, 'Source Pair': px_src
                    })
                    fig_interactive = px.scatter(plotly_df, x='X (norm)', y='Y (norm)', color='Magnitude',
                                                                 color_continuous_scale='viridis', hover_data=['dx', 'dy', 'dz', 'Source Pair'])
                    fig_interactive.update_yaxes(autorange="reversed")
                    fig_interactive.update_layout(height=600)
                    st.plotly_chart(fig_interactive, use_container_width=True)
                except ImportError:
                    st.info("💡 Install `plotly` (`pip install plotly`) to enable interactive hover drilldowns of the deformation field.")'''
                    
text = re.sub(r'                with colB:.*?(?=                # --- Python Plot Script Generation ---)', replace_3 + '\n', text, flags=re.DOTALL)


# 3. Inject 4-panel qc_warping_spatial payload
replace_4 = r'''                # --- Python Plot Script Generation ---
                script_path = os.path.join(analysis_dir, "plot_warping.py")
                csv_path = os.path.join(analysis_dir, "warping_spatial_data.csv")
                
                # Save CSV for script
                import pandas as pd
                pd.DataFrame({
                    'magnitude_px': mags, 
                    'dx': v_dx,
                    'dy': v_dy,
                    'dz': v_dz,
                    'x_norm': x_norm, 
                    'y_norm': y_norm,
                    'source_tile': source_tile
                }).to_csv(csv_path, index=False)
                
                plot_script_content = f"""#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("warping_spatial_data.csv")
mags = df['magnitude_px'].values
x_norm = df['x_norm'].values
y_norm = df['y_norm'].values
v_dx = df['dx'].values
v_dy = df['dy'].values
v_dz = df['dz'].values

plt.rcParams.update({{'font.size': 8}})
fig = plt.figure(figsize=(7.08, 7))

# Plot 1: Histogram
ax1 = plt.subplot(2, 2, 1)
ax1.hist(mags, bins=50, color='skyblue', edgecolor='black')
ax1.set_title("Warping Distribution (Overlap Peripheries)")
ax1.set_xlabel("Displacement Magnitude (pixels)")
ax1.set_ylabel("Frequency (Subsampled Voxels)")
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_yscale('log')

# Plot 2: 2D Spatial Heatmap (Median Magnitude)
ax2a = plt.subplot(2, 2, 2)
hb1 = ax2a.hexbin(x_norm, y_norm, C=mags, gridsize=30, cmap='viridis', 
               reduce_C_function=np.median, mincnt=1)
fig.colorbar(hb1, ax=ax2a, label='Median Displ. (px)')
ax2a.set_title("Universal Tile Overlap Heatmap")
ax2a.set_xlabel("Normalized X Coordinate")
ax2a.set_ylabel("Normalized Y Coordinate")
ax2a.set_xlim(0, 1)
ax2a.set_ylim(0, 1)
ax2a.invert_yaxis()
ax2a.set_aspect('equal', adjustable='box')
ax2a.grid(True, linestyle='--', alpha=0.3)

# Plot 3: Violin Plots
ax3 = plt.subplot(2, 2, 3)
parts = ax3.violinplot([v_dx, v_dy, v_dz], showmeans=False, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(['dx', 'dy', 'dz'])
ax3.set_ylabel("Displacement (pixels)")
ax3.set_title("Deformation Vector Distribution")
ax3.grid(True, linestyle='--', alpha=0.3)

# Plot 4: 2D Density Heatmap (Defensibility)
ax2b = plt.subplot(2, 2, 4)
hb2 = ax2b.hexbin(x_norm, y_norm, gridsize=30, cmap='magma', mincnt=1)
fig.colorbar(hb2, ax=ax2b, label='Vector Count')
ax2b.set_title("Vector Density (Defensibility)")
ax2b.set_xlabel("Normalized X Coordinate")
ax2b.set_ylabel("Normalized Y Coordinate")
ax2b.set_xlim(0, 1)
ax2b.set_ylim(0, 1)
ax2b.invert_yaxis()
ax2b.set_aspect('equal', adjustable='box')
ax2b.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
fig.savefig("qc_warping_spatial.pdf", format='pdf', bbox_inches='tight')
print("Saved qc_warping_spatial.pdf")
plt.show()
"""
                with open(script_path, "w", encoding='utf-8') as f:
                    f.write(plot_script_content)'''
                    
text = re.sub(r'                # --- Python Plot Script Generation ---.*?(?=                \n                st\.info\(f"💾)', replace_4, text, flags=re.DOTALL)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Warping Analysis Injection complete.")
