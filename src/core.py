from enum import Enum
import re
from typing import Tuple, Optional, List, Dict
import os
from dataclasses import dataclass

class ScanOrder(Enum):
    COL_SERPENTINE = "Column Serpentine (pan-ASLM)"
    ROW_SERPENTINE = "Row Serpentine (Boustrophedon)"
    RASTER = "Raster (Row-by-Row)"

class ChannelOrder(Enum):
    INTERLEAVED_PER_Z = "interleaved_per_z"

@dataclass
class DatasetManifest:
    dataset_name: str
    n_tiles_x: int
    n_tiles_y: int
    z_slices: int
    n_channels: int
    overlap_x: int  # Changed to int
    overlap_y: int  # Changed to int
    voxel_size_x_um: float
    voxel_size_y_um: float
    voxel_size_z_um: float
    scan_order: str
    channel_order: str
    width_px: int
    height_px: int
    bit_depth: int
    prefix_filter: str
    files: List[str]

def generate_local_script(manifest: DatasetManifest, output_dir: str):
    """
    Generates a local execution script (run_local.bat for Windows and run_local.sh for POSIX).
    """
    # Windows .bat
    bat_content = f"""@echo off
echo Starting local stitching for {manifest.dataset_name}
echo Date: %DATE% %TIME%

REM Activate your environment here if needed
REM call conda activate pi2_env

echo Running pi2 / NRStitcher...
REM PLACEHOLDER: Please confirm exact pi2 entrypoint
REM python -m pi2.stitch --config stitch_settings.txt --output stitched_output

echo Done.
pause
"""
    with open(os.path.join(output_dir, "run_local.bat"), 'w') as f:
        f.write(bat_content)

    # POSIX .sh
    sh_content = f"""#!/bin/bash
echo "Starting local stitching for {manifest.dataset_name}"
date

# source activate pi2_env

echo "Running pi2 / NRStitcher..."
# PLACEHOLDER: Please confirm exact pi2 entrypoint
# python -m pi2.stitch --config stitch_settings.txt --output stitched_output

echo "Done"
date
"""
    with open(os.path.join(output_dir, "run_local.sh"), 'w') as f:
        f.write(sh_content)
        # Make executable
        try:
            os.chmod(os.path.join(output_dir, "run_local.sh"), 0o755)
        except:
            pass # Windows might fail on chmod if not careful


def map_index(i: int, n_channels: int, z_slices: int) -> Tuple[int, int, int]:
    """
    Maps linear index to (tile_idx, z_idx, ch_idx) for interleaved-per-Z ordering.
    
    Ordering:
    For a given Tile:
      Z1: C1, C2, ... CC
      Z2: C1, C2, ... CC
      ...
    
    So the inner-most loop is Channels, then Z, then Tiles.
    
    Args:
        i: Linear index (0-based)
        n_channels: Number of channels
        z_slices: Number of Z slices
        
    Returns:
        (tile_idx, z_idx, ch_idx)
    """
    total_images_per_tile = n_channels * z_slices
    
    tile_idx = i // total_images_per_tile
    remainder = i % total_images_per_tile
    
    z_idx = remainder // n_channels
    ch_idx = remainder % n_channels
    
    return tile_idx, z_idx, ch_idx

def tile_idx_to_xy(tile_idx: int, n_tiles_x: int, n_tiles_y: int, scan_order: str) -> Tuple[int, int]:
    """
    Maps linear tile index to (x, y) grid coordinates.
    
    Supports three scan orders:
    - Column Serpentine (pan-ASLM): X slow, Y fast. Even cols go up, odd cols go down.
    - Row Serpentine (Boustrophedon): Y slow, X fast. Even rows go right, odd rows go left.
    - Raster: Y slow, X fast. All rows go left to right.
    
    Args:
        tile_idx: Linear tile index
        n_tiles_x: Number of tiles in X (columns)
        n_tiles_y: Number of tiles in Y (rows)
        scan_order: One of the ScanOrder enum values
        
    Returns:
        (x, y) grid coordinates
    """
    if scan_order == ScanOrder.COL_SERPENTINE.value:
        # Column-wise serpentine: X slow, Y fast
        x = tile_idx // n_tiles_y
        y_raw = tile_idx % n_tiles_y
        if x % 2 == 0:
            y = y_raw  # Even column: y ascending (up)
        else:
            y = (n_tiles_y - 1) - y_raw  # Odd column: y descending (down)
    elif scan_order == ScanOrder.ROW_SERPENTINE.value:
        # Row-wise serpentine: Y slow, X fast
        y = tile_idx // n_tiles_x
        x_raw = tile_idx % n_tiles_x
        if y % 2 == 0:
            x = x_raw  # Even row: x ascending (right)
        else:
            x = (n_tiles_x - 1) - x_raw  # Odd row: x descending (left)
    else:
        # Raster: row-by-row, left to right
        y = tile_idx // n_tiles_x
        x = tile_idx % n_tiles_x
    
    return x, y


def xy_to_tile_idx(gx: int, gy: int, n_tiles_x: int, n_tiles_y: int, scan_order: str) -> int:
    """
    Reverse of tile_idx_to_xy: converts (x, y) grid coordinates to linear tile index.
    """
    if scan_order == ScanOrder.COL_SERPENTINE.value:
        if gx % 2 == 0:
            return gx * n_tiles_y + gy
        else:
            return gx * n_tiles_y + (n_tiles_y - 1 - gy)
    elif scan_order == ScanOrder.ROW_SERPENTINE.value:
        if gy % 2 == 0:
            return gy * n_tiles_x + gx
        else:
            return gy * n_tiles_x + (n_tiles_x - 1 - gx)
    else:
        return gy * n_tiles_x + gx

def parse_filename(filename: str) -> Optional[int]:
    """
    Extracts the numeric suffix from a filename.
    Expects format like 'prefix_00001.tif'.
    Returns None if no digits are found at the end of the stem.
    """
    # Remove extension
    stem = os.path.splitext(filename)[0]
    
    # Find all digits at the end of the string
    match = re.search(r'(\d+)$', stem)
    if match:
        return int(match.group(1))
    return None

def load_files(data_path: str, prefix_filter: str = "") -> List[str]:
    """
    Scans directory for .tif/.tiff files, optionally filtering by prefix.
    Returns sorted list of filenames.
    """
    if not os.path.isdir(data_path):
        # Return empty if path doesn't exist, or raise? Raised in previous version.
        # Streamlit might prefer no error, just empty.
        # But core logic should probably raise or return empty.
        # Let's return empty to be safe for UI.
        return []
        
    files = []
    try:
        for f in os.listdir(data_path):
            if f.lower().endswith(('.tif', '.tiff')):
                if prefix_filter and not f.startswith(prefix_filter):
                    continue
                files.append(f)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []
            
    # Sort files naturally/lexicographically to give a baseline validation order
    files.sort()
    return files

def validate_dataset(files: List[str]) -> Dict:
    """
    Validates the dataset files.
    Checks for:
    - Non-empty file list
    - Suffix continuity (gaps, missing files)
    
    Returns a dict with:
        'valid': bool
        'message': str
        'n_files': int
        'min_idx': int
        'max_idx': int
        'missing_indices': List[int]
    """
    if not files:
        return {'valid': False, 'message': "No files found", 'n_files': 0}
        
    indices = []
    for f in files:
        idx = parse_filename(f)
        if idx is not None:
            indices.append(idx)
            
    if not indices:
        return {'valid': False, 'message': "Could not parse numeric suffixes from filenames", 'n_files': len(files)}
        
    indices.sort()
    min_idx = indices[0]
    max_idx = indices[-1]
    expected_count = max_idx - min_idx + 1
    
    missing = []
    # Check for gaps if we don't have the expected number of files
    if len(indices) != expected_count:
        idx_set = set(indices)
        for i in range(min_idx, max_idx + 1):
            if i not in idx_set:
                missing.append(i)
                if len(missing) > 10: # limit reporting
                    break
                    
    if missing:
        msg = f"Found {len(indices)} files, but index range {min_idx}-{max_idx} implies {expected_count}. Missing {len(missing)} files (e.g., {missing[:5]}...)"
        return {
            'valid': False,
            'message': msg,
            'n_files': len(files),
            'min_idx': min_idx,
            'max_idx': max_idx,
            'missing_indices': missing
        }
        
    return {
        'valid': True, 
        'message': f"Found {len(files)} files with continuous indices {min_idx}-{max_idx}",
        'n_files': len(files),
        'min_idx': min_idx,
        'max_idx': max_idx,
        'missing_indices': []
    }

def infer_tiff_metadata(file_path: str) -> Tuple[int, int, int]:
    """
    Reads the first file to infer width, height, and bit depth.
    Uses tifffile.
    """
    import tifffile
    with tifffile.TiffFile(file_path) as tif:
        page = tif.pages[0]
        # shape is usually (height, width) or (depth, height, width)
        # We assume 2D slices based on spec
        h, w = page.shape[-2:] if len(page.shape) >= 2 else page.shape
        dtype = page.dtype
        
        # Estimate bit depth from dtype
        if dtype.itemsize == 1:
            bit_depth = 8
        elif dtype.itemsize == 2:
            bit_depth = 16
        elif dtype.itemsize == 4:
            bit_depth = 32
        else:
            bit_depth = 0 # Unknown
            
        return w, h, bit_depth

def generate_stitch_settings(manifest: DatasetManifest, output_dir: str, data_path: str, use_tiles_view: bool, stitch_output_format: str = "raw"):
    """
    Generates stitch_settings.txt in INI format for pi2/NRStitcher.
    Uses [stitch] section for parameters and [positions] for tile file paths.
    Points to pre-stacked 3D tile files in stacks/ directory.
    """
    
    # Calculate step sizes in pixels
    overlap_x_frac = manifest.overlap_x / 100.0
    overlap_y_frac = manifest.overlap_y / 100.0
    # User requested 1 sig fig (likely 1 decimal place) for coordinates
    step_x = manifest.width_px * (1 - overlap_x_frac)
    step_y = manifest.height_px * (1 - overlap_y_frac)
    
    n_tiles = manifest.n_tiles_x * manifest.n_tiles_y
    
    lines = []
    # Comments use ; in INI format
    lines.append(f"; Stitch Settings for {manifest.dataset_name}")
    lines.append(f"; Generated by Antigravity Run Bundle Generator")
    lines.append("")
    
    # [stitch] section - stitching parameters
    lines.append("[stitch]")
    lines.append(f"sample_name = {manifest.dataset_name}")
    lines.append("binning = 1")
    lines.append("point_spacing = 20")
    lines.append("coarse_block_radius = [25, 25, 25]")
    lines.append("coarse_binning = 4")
    lines.append("fine_block_radius = [25, 25, 25]")
    lines.append("fine_binning = 1")
    lines.append("normalize_in_blockmatch = True")
    lines.append("normalize_while_stitching = True")
    lines.append("global_optimization = True")
    lines.append("allow_rotation = True")
    lines.append("allow_local_deformations = True")
    lines.append("zeroes_are_missing_values = True")
    lines.append(f"output_format = {stitch_output_format}")
    lines.append("")
    
    # [positions] section - one entry per tile: filepath = x, y, z
    lines.append("[positions]")
    
    for t in range(n_tiles):
        # Calculate grid coordinates using serpentine pattern
        grid_x, grid_y = tile_idx_to_xy(t, manifest.n_tiles_x, manifest.n_tiles_y, manifest.scan_order)
        
        # Pixel positions in the stitched coordinate space
        pos_x = grid_x * step_x
        pos_y = grid_y * step_y
        pos_z = 0  # 2D grid of 3D stacks; all start at Z=0
        
        # Point to the pre-stacked 3D tile files created by stack_tiles.py
        if manifest.n_channels > 1:
            filename = f"stacks/tile_{t:03d}_ch0.tif"
        else:
            filename = f"stacks/tile_{t:03d}.tif"
        
        # Minimal formatting: Integer if possible, else 1 decimal place
        def fmt(val):
            if isinstance(val, int):
                return str(val)
            return f"{val:.0f}" if val.is_integer() else f"{val:.1f}"
            
        lines.append(f"{filename} = {fmt(pos_x)}, {fmt(pos_y)}, {fmt(pos_z)}")
    
    content = "\n".join(lines)
    
    with open(os.path.join(output_dir, "stitch_settings.txt"), 'w') as f:
        f.write(content)

def generate_slurm_script(manifest: DatasetManifest, slurm_params: Dict, output_dir: str):
    """
    Generates run_nrstitcher.sbatch targeting pi2/NRStitcher on Misha.
    """
    script = f"""#!/bin/bash
#SBATCH --job-name={manifest.dataset_name}_stitch
#SBATCH --partition={slurm_params.get('partition', 'day')}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_params.get('cpus', 8)}
#SBATCH --mem={slurm_params.get('mem', '64G')}
#SBATCH --time={slurm_params.get('time', '04:00:00')}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Starting stitching job for {manifest.dataset_name}"
date

# Load environment (template)
# module load miniconda
# conda activate pi2_env
# OR
# apptainer exec /path/to/pi2_container.sif python -m pi2.stitch ...

# PLACEHOLDER: Please confirm exact pi2 entrypoint
echo "Running pi2 / NRStitcher..."
# srun python -m pi2.stitch --config stitch_settings.txt --output stitched_output

echo "Done"
date
"""
    with open(os.path.join(output_dir, "run_nrstitcher.sbatch"), 'w') as f:
        f.write(script)

def generate_manifest(manifest: DatasetManifest, output_dir: str):
    """
    Writes the dataset manifest to JSON.
    """
    import json
    # Convert Dataclass to dict
    # Filter files list to avoid huge JSON if too large? 
    # Validated files list is good to have but maybe large. 
    # We will include it for reproducibility as requested.
    
    data = manifest.__dict__.copy()
    
    # Write to file
    with open(os.path.join(output_dir, "dataset_manifest.json"), 'w') as f:
        json.dump(data, f, indent=2)

def generate_tiles_view(manifest: DatasetManifest, output_dir: str, data_path: str):
    """
    Creates a 'tiles' directory with symlinks to original files, renamed clearly.
    Format: tile_{t:04d}_z_{z:04d}_c_{c:02d}.tif
    """
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    success_count = 0
    errors = []
    
    for i, filename in enumerate(manifest.files):
        src_path = os.path.join(data_path, filename)
        if not os.path.exists(src_path):
            errors.append(f"Source missing: {filename}")
            continue
            
        tile_idx, z_idx, ch_idx = map_index(i, manifest.n_channels, manifest.z_slices)
        
        # Determine extension
        _, ext = os.path.splitext(filename)
        new_name = f"tile_{tile_idx:04d}_z_{z_idx:04d}_c_{ch_idx:02d}{ext}"
        dst_path = os.path.join(tiles_dir, new_name)
        
        try:
            # Remove existing if any
            if os.path.exists(dst_path):
                os.remove(dst_path)
            
            # Create symlink
            # Windows: os.symlink(src, dst) requires Src to be absolute? Yes usually.
            # And requires Privileges.
            os.symlink(os.path.abspath(src_path), dst_path)
            success_count += 1
        except OSError as e:
            # Fallback? Hardlink?
            # On Windows, error 1314 is "A required privilege is not held by the client".
            # Check for winerror attribute which is specific to Windows OSError
            if hasattr(e, 'winerror') and e.winerror == 1314:
                errors.append("Symlink failed (Permission Denied). Enable Developer Mode or Run as Admin.")
                break # Stop trying
            else:
                errors.append(f"Link failed for {filename}: {e}")
                
    if len(errors) > 0 and success_count == 0:
        raise OSError("\n".join(errors[:5]))
    
    return success_count, errors

def generate_tiles_view(manifest: DatasetManifest, output_dir: str):
    """
    Creates a 'tiles' directory with symlinks to original files, renamed clearly.
    Format: tile_{t:04d}_z_{z:04d}_c_{c:02d}.tif
    """
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    # We assume data_path (source files) are where manifest.files point to.
    # But manifest.files are basenames.
    # We need the source directory.
    # The manifest doesn't explicitly store the absolute source dir, just basenames?
    # Wait, `load_files` returned basenames.
    # We need the source path passed in or we rely on the user running this from the app 
    # where we had `data_path`.
    # Let's check DatasetManifest... it has `files`, but no `source_dir`.
    # We should add `source_dir` to Manifest or pass it here?
    # Passing it here is flexible but Manifest should probably be self-contained for reproducibility.
    # Let's pass `source_dir` as argument for now, assuming it's available in the App.
    pass 
    # Wait, I can't modify the signature of the call in App without updating Manifest?
    # App calls: generate_manifest(manifest, output_dir)
    # App has `data_path`.
    
    # Let's look at `generate_tiles_view` signature in my plan: `generate_tiles_view(manifest, output_dir)`.
    # I need `data_path` to resolve absolute paths for symlinks.
    # I will modify the function signature to accept `data_path` or add it to Manifest.
    # Adding to Manifest is cleaner for "re-running" later. 
    # But I'll modify the function signature for immediate fix.
    
    raise  NotImplementedError("Need data_path") 

# Redefining to accept data_path
def generate_tiles_view(manifest: DatasetManifest, output_dir: str, data_path: str):
    """
    Creates a 'tiles' directory with symlinks to original files, renamed clearly.
    Format: tile_{t:04d}_z_{z:04d}_c_{c:02d}.tif
    """
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    success_count = 0
    errors = []
    
    for i, filename in enumerate(manifest.files):
        src_path = os.path.join(data_path, filename)
        if not os.path.exists(src_path):
            errors.append(f"Source missing: {filename}")
            continue
            
        tile_idx, z_idx, ch_idx = map_index(i, manifest.n_channels, manifest.z_slices)
        
        # Determine extension
        _, ext = os.path.splitext(filename)
        new_name = f"tile_{tile_idx:04d}_z_{z_idx:04d}_c_{ch_idx:02d}{ext}"
        dst_path = os.path.join(tiles_dir, new_name)
        
        try:
            # Remove existing if any
            if os.path.exists(dst_path):
                os.remove(dst_path)
            
            # Create symlink
            # Windows: os.symlink(src, dst) requires Src to be absolute? Yes usually.
            # And requires Privileges.
            os.symlink(os.path.abspath(src_path), dst_path)
            success_count += 1
        except OSError as e:
            # Fallback? Hardlink?
            # On Windows, error 1314 is "A required privilege is not held by the client".
            if hasattr(e, 'winerror') and e.winerror == 1314:
                errors.append("Symlink failed (Permission Denied). Enable Developer Mode or Run as Admin.")
                break # Stop trying
            else:
                errors.append(f"Link failed for {filename}: {e}")
                
    if len(errors) > 0 and success_count == 0:
        raise OSError("\n".join(errors[:5]))
    
    return success_count, errors


def generate_local_script(manifest: DatasetManifest, output_dir: str, conda_config: Dict[str, str], embed_pi2_path: Optional[str] = None, convert_ometiff: bool = False):
    """
    Generates a local execution script (run_local.bat for Windows and run_local.sh for POSIX).
    Optionally embeds a copy of 'pi2' package into 'tools/' and sets PYTHONPATH.
    """
    conda_sh = conda_config.get('conda_sh', '')
    env_name = conda_config.get('env_name', 'pi2_env')
    entrypoint_override = conda_config.get('entrypoint', '') # Optional

    # Embedding Logic
    pythonpath_env_var_win = ""
    pythonpath_env_var_sh = ""
    target_subdir = "pi2" 
    is_binary_dist = False
    
    if embed_pi2_path and os.path.isdir(embed_pi2_path):
        import shutil
        tools_dir = os.path.join(output_dir, "tools")
        
        # Check if this is the flat binary distro (has nr_stitcher.py but no __init__.py usually)
        if os.path.exists(os.path.join(embed_pi2_path, "nr_stitcher.py")):
            is_binary_dist = True
            target_subdir = "pi2_dist"
            
        target_pi2 = os.path.join(tools_dir, target_subdir)
        
        # Clean previous tools if exists (to update)
        if os.path.exists(target_pi2):
             try:
                 shutil.rmtree(target_pi2)
             except:
                 pass
        
        try:
            # Copy specific package
            shutil.copytree(embed_pi2_path, target_pi2, ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git', '.idea', '.vscode'))
            
            # Update PYTHONPATH so we can import things from this folder
            pythonpath_env_var_win = f"set PYTHONPATH=%~dp0tools\\{target_subdir};%PYTHONPATH%"
            pythonpath_env_var_sh = f'export PYTHONPATH="$(dirname "$0")/tools/{target_subdir}:$PYTHONPATH"'
            
        except Exception as e:
            print(f"Failed to embed pi2: {e}") 

    # Determine Command
    if entrypoint_override:
        cmd_win = entrypoint_override
    elif is_binary_dist:
        # For binary dist, we usually run nr_stitcher.py directly
        cmd_win = f"python \"%~dp0tools\\{target_subdir}\\nr_stitcher.py\""
    else:
        # Default python package assume
        cmd_win = "python -m pi2.stitch"

    # Determine Arguments
    if is_binary_dist:
        stitch_args = "stitch_settings.txt"
    else:
        # Default behavior for python module execution (if different)
        stitch_args = f"--config stitch_settings.txt --output {manifest.dataset_name}_stitched"
    
    bat_content = f"""@echo off
cd /d "%~dp0"
echo Starting local stitching for {manifest.dataset_name}
echo Working Dir: %CD%
echo Date: %DATE% %TIME%

{pythonpath_env_var_win}

REM Activate environment
echo Activating {env_name}...
if exist "{conda_sh}" goto UseCondaPath
call conda activate {env_name}
goto CheckActivate

:UseCondaPath
call "{conda_sh}" activate {env_name}

:CheckActivate
if not errorlevel 1 goto CheckDeps

:CreateEnv
echo [WARN] Could not activate environment '{env_name}'. Attempting to create it...
if exist "{conda_sh}" goto CreateCondaPath
call conda create -n {env_name} python=3.9 -y
call conda activate {env_name}
goto CheckCreate

:CreateCondaPath
call "{conda_sh}" create -n {env_name} python=3.9 -y
call "{conda_sh}" activate {env_name}

:CheckCreate
if not errorlevel 1 goto InstallDeps
echo [ERROR] Failed to create/activate environment. Exiting.
pause
exit /b 1

:CheckDeps
python -c "import networkx; import pyquaternion; import scipy" 2>NUL
if not errorlevel 1 goto ActivationDone

:InstallDeps
echo [INFO] Installing/Updating dependencies (networkx, pyquaternion, scipy)...
pip install numpy tifffile scikit-image networkx pyquaternion scipy

:ActivationDone
REM Auto-install/Path check
python -c "import pi2py2" 2>NUL
if not errorlevel 1 goto StackTiles

python -c "import pi2" 2>NUL
if not errorlevel 1 goto StackTiles

:CheckEmbedded
echo [INFO] 'pi2' or 'pi2py2' module not found in environment.
if exist "%~dp0tools\\{target_subdir}" goto FoundEmbedded

echo [ERROR] 'pi2' missing and cannot be auto-installed (requires C++ binaries).
echo [ACTION] Please download the Windows binary from: https://github.com/arttumiettinen/pi2/releases
echo [ACTION] Extract it, and set the "Path to 'pi2'" in the App to that folder.
pause
exit /b 1

:FoundEmbedded
echo [INFO] Found embedded tools in tools\\{target_subdir}. Setting PYTHONPATH should fix this.

:StackTiles
REM === Fast-path: skip stacking if stacks already exist ===
set "STACK_COUNT=0"
for /f %%A in ('dir /b stacks\*.tif 2^>NUL ^| find /c /v ""') do set "STACK_COUNT=%%A"
if %STACK_COUNT% GTR 0 echo [INFO] Found %STACK_COUNT% existing stack(s) in stacks. Skipping stacking.
if %STACK_COUNT% GTR 0 echo [INFO] To force re-stacking, delete the stacks folder first.
if %STACK_COUNT% GTR 0 goto VerifyStacks

echo.
echo ============================================================
echo   STACKING: Compiling 2D slices into 3D volumes
echo ============================================================
python stack_tiles.py
if errorlevel 1 goto StackFailed
goto VerifyStacks

:StackFailed
echo [ERROR] Stacking failed!
pause
exit /b 1

:VerifyStacks
REM === Post-stack verification ===
echo.
echo --- Stack Verification ---
echo   Path: %~dp0stacks
set "STACK_COUNT=0"
for /f %%A in ('dir /b stacks\*.tif 2^>NUL ^| find /c /v ""') do set "STACK_COUNT=%%A"
echo   Stack count: %STACK_COUNT% .tif file(s)
if %STACK_COUNT% EQU 0 goto StacksEmpty
for %%F in (stacks\tile_*.tif) do echo   First stack: %%~nxF (%%~zF bytes)& goto DoneVerify
:DoneVerify
echo --- Verification OK ---
echo.
goto RunStitcher

:StacksEmpty
echo [ERROR] stacks\\ not found or empty; stacking step was skipped or failed.
echo [ACTION] Delete the stacks folder and re-run, or check stack_tiles.py output above.
pause
exit /b 1

:RunStitcher
echo ============================================================
echo   STITCHING: Running pi2 / NRStitcher
echo ============================================================
echo Command: {cmd_win} stitch_settings.txt
{cmd_win} stitch_settings.txt
if errorlevel 1 goto StitchFailed

{'''echo.
echo ============================================================
echo   CONVERTING: Raw output to OME-TIFF
echo ============================================================
python convert_to_ometiff.py
if errorlevel 1 echo [WARN] OME-TIFF conversion failed. Raw output should still exist.
''' if convert_ometiff else ''}

echo.
echo ============================================================
echo   CLEANUP: Moving debug logs to trace/
echo ============================================================
if not exist trace mkdir trace
move *defpoints* trace\ >nul 2>&1
move *refpoints* trace\ >nul 2>&1
move *gof* trace\ >nul 2>&1
move *transformation.txt trace\ >nul 2>&1
move *global_positions.txt trace\ >nul 2>&1
move *_done.tif trace\ >nul 2>&1
goto AllDone

:StitchFailed
echo [ERROR] Stitching failed!
pause
exit /b 1

:AllDone
echo.
echo ============================================================
echo   ALL DONE
echo ============================================================
pause
"""
    with open(os.path.join(output_dir, "run_local.bat"), 'w') as f:
        f.write(bat_content)

    # Bash detection logic
    detection_block = f"""
# 3. Auto-detect entrypoint
STITCH_CMD=""
USER_OVERRIDE="{entrypoint_override}"

if [ ! -z "$USER_OVERRIDE" ]; then
    echo "Using user override: $USER_OVERRIDE"
    STITCH_CMD="$USER_OVERRIDE"
elif [ "{is_binary_dist}" = "True" ]; then
    STITCH_CMD="python $(dirname "$0")/tools/{target_subdir}/nr_stitcher.py"
elif command -v nrstitcher &> /dev/null; then
    STITCH_CMD="nrstitcher"
elif command -v pi2 &> /dev/null; then
    STITCH_CMD="pi2 stitch"
elif python -m pi2.stitch -h &> /dev/null; then
    STITCH_CMD="python -m pi2.stitch"
else
    # Try local binary dist direct check
    if [ -f "$(dirname "$0")/tools/{target_subdir}/nr_stitcher.py" ]; then
         STITCH_CMD="python $(dirname "$0")/tools/{target_subdir}/nr_stitcher.py"
    else
        echo "Error: Could not find 'nrstitcher', 'pi2', or embedded tools."
        exit 1
    fi
fi

echo "Using stitch command: $STITCH_CMD"

# Preflight
if $STITCH_CMD --version &> /dev/null; then
    $STITCH_CMD --version
elif $STITCH_CMD -h &> /dev/null; then
    echo "Verified ($STITCH_CMD -h works)."
else
    echo "Error: '$STITCH_CMD' found but seems broken (failed --version and -h)."
    exit 1
fi
"""

    sh_content = f"""#!/bin/bash
# Ensure we are in the script directory
cd "$(dirname "$0")"

echo "Starting local stitching for {manifest.dataset_name}"
echo "Working Dir: $(pwd)"
date

{pythonpath_env_var_sh}

# Initialize Conda
if [ -f "{conda_sh}" ]; then
    source "{conda_sh}"
elif [ ! -z "$CONDA_EXE" ]; then
    # Try to derive from current env if running locally
    # Hook usually in shell, but explicit source is safer
    echo "Warning: Conda init script not specified. Relying on current shell or PATH."
fi

echo "Activating {env_name}..."
conda activate {env_name}

if [ $? -ne 0 ]; then
    echo "[WARN] Could not activate environment '{env_name}'."
    echo "Attempting to create it..."
    conda create -n {env_name} python=3.9 -y
    conda activate {env_name}
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create/activate environment. Exiting."
        exit 1
    fi
# Check dependencies
python -c "import networkx; import pyquaternion; import scipy" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing dependencies..."
    pip install numpy tifffile scikit-image networkx pyquaternion scipy
fi

# Auto-install/Path check
if ! python -c "import pi2py2" &> /dev/null; then
    if ! python -c "import pi2" &> /dev/null; then
        echo "[INFO] 'pi2/pi2py2' module not found."
        if [ -d "$(dirname "$0")/tools/{target_subdir}" ]; then
            echo "[INFO] Found embedded tools. Using them."
        else
            echo "[ERROR] 'pi2' missing and cannot be auto-installed (requires C++ binaries)."
            echo "[ACTION] Please download binaries."
            exit 1
        fi
    fi
fi

{detection_block}

# === Fast-path: skip stacking if stacks already exist and are valid ===
STACK_COUNT=$(find stacks -maxdepth 1 -name '*.tif' 2>/dev/null | wc -l)
if [ "$STACK_COUNT" -gt 0 ]; then
    echo "[INFO] Found $STACK_COUNT existing stack(s) in stacks/. Skipping stacking step."
    echo "[INFO] To force re-stacking, delete the stacks/ folder first."
else
    echo ""
    echo "============================================================"
    echo "  STACKING: Compiling 2D slices into 3D volumes"
    echo "============================================================"
    python stack_tiles.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Stacking failed!"
        exit 1
    fi
fi

# === Post-stack verification ===
echo ""
echo "--- Stack Verification ---"
STACKS_DIR="$(pwd)/stacks"
echo "  Path: $STACKS_DIR"
STACK_COUNT=$(find stacks -maxdepth 1 -name '*.tif' 2>/dev/null | wc -l)
echo "  Stack count: $STACK_COUNT .tif file(s)"

if [ "$STACK_COUNT" -eq 0 ]; then
    echo "[ERROR] stacks/ not found or empty; stacking step was skipped or failed."
    echo "[ACTION] Delete the stacks/ folder and re-run, or check stack_tiles.py output above."
    exit 1
fi

FIRST_STACK=$(ls stacks/*.tif 2>/dev/null | head -1)
if [ -n "$FIRST_STACK" ]; then
    FSIZE=$(stat --printf="%s" "$FIRST_STACK" 2>/dev/null || stat -f%z "$FIRST_STACK" 2>/dev/null || echo "unknown")
    echo "  First stack: $FIRST_STACK ($FSIZE bytes)"
fi
echo "--- Verification OK ---"
echo ""

echo "============================================================"
echo "  STITCHING: Running pi2 / NRStitcher"
echo "============================================================"
echo "Command: $STITCH_CMD {stitch_args}"
$STITCH_CMD {stitch_args}
if [ $? -ne 0 ]; then
    echo "[ERROR] Stitching failed!"
    exit 1
fi

{"echo" + chr(10) + 'echo "============================================================"' + chr(10) + 'echo "  CONVERTING: Raw output to OME-TIFF"' + chr(10) + 'echo "============================================================"' + chr(10) + "python convert_to_ometiff.py" + chr(10) + 'if [ $? -ne 0 ]; then' + chr(10) + '    echo "WARNING: OME-TIFF conversion failed. Raw output should still exist."' + chr(10) + "fi" if convert_ometiff else ""}

echo ""
echo "============================================================"
echo "  CLEANUP: Moving debug logs to trace/"
echo "============================================================"
mkdir -p trace
mv *defpoints* *refpoints* *gof* trace/ 2>/dev/null
mv *transformation.txt *global_positions.txt trace/ 2>/dev/null
mv *_done.tif trace/ 2>/dev/null

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "============================================================"
date
"""
    with open(os.path.join(output_dir, "run_local.sh"), 'w') as f:
        f.write(sh_content)
        try:
            os.chmod(os.path.join(output_dir, "run_local.sh"), 0o755)
        except:
            pass


def generate_slurm_script(manifest: DatasetManifest, slurm_params: Dict, output_dir: str, conda_config: Dict[str, str]):
    """
    Generates run_nrstitcher.sbatch targeting pi2/NRStitcher on Misha.
    """
    conda_sh = conda_config.get('conda_sh', '~/miniconda3/etc/profile.d/conda.sh')
    env_name = conda_config.get('env_name', 'pi2_env')
    entrypoint_override = conda_config.get('entrypoint', '')

    detection_block = f"""
# 3. Auto-detect entrypoint
STITCH_CMD=""
USER_OVERRIDE="{entrypoint_override}"

if [ ! -z "$USER_OVERRIDE" ]; then
    echo "Using user override: $USER_OVERRIDE"
    STITCH_CMD="$USER_OVERRIDE"
elif command -v nrstitcher &> /dev/null; then
    STITCH_CMD="nrstitcher"
elif command -v pi2 &> /dev/null; then
    STITCH_CMD="pi2 stitch"
elif python -m pi2.stitch -h &> /dev/null; then
    STITCH_CMD="python -m pi2.stitch"
else
    echo "Error: Could not find 'nrstitcher' or 'pi2' on PATH, and 'python -m pi2.stitch -h' failed."
    echo "Action: Either (a) install pi2 into this conda env, or (b) set Entrypoint Override to the correct command."
    exit 1
fi

echo "Using stitch command: $STITCH_CMD"

# Preflight
if $STITCH_CMD --version &> /dev/null; then
    $STITCH_CMD --version
elif $STITCH_CMD -h &> /dev/null; then
    echo "Verified ($STITCH_CMD -h works)."
else
    echo "Error: '$STITCH_CMD' found but seems broken (failed --version and -h)."
    exit 1
fi
"""

    script = f"""#!/bin/bash
#SBATCH --job-name={manifest.dataset_name}_stitch
#SBATCH --partition={slurm_params.get('partition', 'day')}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_params.get('cpus', 8)}
#SBATCH --mem={slurm_params.get('mem', '64G')}
#SBATCH --time={slurm_params.get('time', '04:00:00')}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Starting stitching job for {manifest.dataset_name}"
date
echo "Host: $(hostname)"

# 1. Initialize Conda
# Try to auto-derive if path not robust, but manual path is best for Slurm
CONDA_INIT="{conda_sh}"
if [ -f "$CONDA_INIT" ]; then
    echo "Sourcing Conda from $CONDA_INIT"
    source "$CONDA_INIT"
else
    echo "Warning: Conda init script not found at $CONDA_INIT"
    echo "Attempting to derive from 'conda info --base' if available..."
    BASE=$(conda info --base 2>/dev/null)
    if [ ! -z "$BASE" ]; then
         source "$BASE/etc/profile.d/conda.sh"
    else
         echo "Error: Could not source conda. Please check path."
         # Don't exit yet, maybe it's in PATH
    fi
fi

# 2. Activate Environment
echo "Activating environment: {env_name}"
conda activate {env_name}

{detection_block}

# 4. Run Stitcher
echo "Running command: $STITCH_CMD"
srun $STITCH_CMD --config stitch_settings.txt --output {manifest.dataset_name}_stitched

echo "Done"
date
"""
    with open(os.path.join(output_dir, "run_nrstitcher.sbatch"), 'w') as f:
        f.write(script)


def get_tile_preview(file_path: str) -> Tuple[Optional[object], Optional[str], Optional[Dict]]:
    """
    Reads a TIFF file and returns a normalized numpy array for preview.
    Returns: (image_array, error_message, stats_dict)
    """
    try:
        import tifffile
        import numpy as np
        
        # Read the first page/series
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            img = page.asarray()
            
        stats = {
            'orig_min': float(np.min(img)),
            'orig_max': float(np.max(img)),
            'dtype': str(img.dtype),
            'shape': str(img.shape)
        }

        # Normalize for display (robust percentile-based Auto B/C)
        # Convert to float for calculation
        img_f = img.astype(np.float32)
        
        # Robust min/max using percentiles
        low = np.percentile(img_f, 1)
        high = np.percentile(img_f, 99)
        
        # If flat or nearly flat, fall back to min/max
        if high <= low:
            low = np.min(img_f)
            high = np.max(img_f)
            
        if high > low:
            img_n = (img_f - low) / (high - low) * 255.0
            img_n = np.clip(img_n, 0, 255).astype(np.uint8)
        else:
            img_n = np.zeros_like(img, dtype=np.uint8)
                
        return img_n, None, stats
    except Exception as e:
        return None, str(e), None

def generate_stack_script(manifest: DatasetManifest, output_dir: str, data_path: str):
    """
    Generates a Python script 'stack_tiles.py' to convert 2D slices into 3D stacks.
    This is necessary for efficient 3D stitching.
    """
    
    script_content = f"""import os
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
import time
import skimage.io

# Configuration
DATA_PATH = r"{data_path}"
OUTPUT_DIR = "stacks"
FILES = {manifest.files}
N_CHANNELS = {manifest.n_channels}
Z_SLICES = {manifest.z_slices}
N_TILES = {manifest.n_tiles_x * manifest.n_tiles_y}

def parse_filename(filename):
    # Extract number from end
    import re
    match = re.search(r'(\d+)', filename[::-1])
    if match:
        return int(match.group(1)[::-1])
    return None

def map_index(idx, n_channels, z_slices):
    # Same logic as core.py
    # channel -> z -> tile (slowest)
    c_idx = idx % n_channels
    remaining = idx // n_channels
    z_idx = remaining % z_slices
    t_idx = remaining // z_slices
    return t_idx, z_idx, c_idx

def process_tile(t_idx):
    # Find all files for this tile
    # We could iterate all files, but that's slow.
    # We know the indices.
    
    # Filter files belonging to this tile
    # This might be slow if list is huge. 
    # Better: Pre-group.
    pass

def main():
    print("Starting Stacking Process...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group files by tile
    tiles = {{}} # t_idx -> list of (z_idx, c_idx, filename)
    
    print("Grouping files...")
    for i, f in enumerate(FILES):
        t, z, c = map_index(i, N_CHANNELS, Z_SLICES)
        if t not in tiles:
            tiles[t] = []
        tiles[t].append((z, c, f))
        
    print(f"Found {{len(tiles)}} tiles.")
    
    # Process each tile
    for t_idx, items in tiles.items():
        # Sort by Z
        items.sort(key=lambda x: x[0])
        
        # We might have multiple channels.
        # Stitcher usually expects 1 channel or handles them?
        # If we have multiple channels, we usually stack them as (C, Z, Y, X) or separate files?
        # nr_stitcher usually takes one file per location.
        # If we have channels, we often stitch channel 0 and apply transform to others.
        # Let's stack Channel 0 for now as 'tile_T.tif'.
        # If user wants other channels, we might need 'tile_T_chC.tif'.
        
        # Group by channel
        channels = {{}}
        for z, c, f in items:
            if c not in channels:
                channels[c] = []
            channels[c].append((z, f))
            
        for c, z_files in channels.items():
            # Verify Z completeness?
            if len(z_files) != Z_SLICES:
                print(f"Warning: Tile {{t_idx}} Ch {{c}} has {{len(z_files)}} slices, expected {{Z_SLICES}}")
                
            # stack name
            if N_CHANNELS > 1:
                out_name = f"tile_{{t_idx:03d}}_ch{{c}}.tif"
            else:
                out_name = f"tile_{{t_idx:03d}}.tif"
                
            out_path = os.path.join(OUTPUT_DIR, out_name)
            if os.path.exists(out_path):
                print(f"Skipping {{out_name}} (exists)")
                continue
                
            print(f"Stacking {{out_name}}...")
            
            # Read images
            # Lazy approach: read first to allow memory estimation?
            # Or just append?
            # tifffile.imwrite can append?
            # Or just read all into numpy array (memory heavy but faster)
            try:
                # Read first to get shape
                first_f = os.path.join(DATA_PATH, z_files[0][1])
                first_img = skimage.io.imread(first_f)
                dtype = first_img.dtype
                shape = first_img.shape # (Y, X)
                
                # Pre-allocate volume (Z, Y, X)
                vol = np.zeros((Z_SLICES, shape[0], shape[1]), dtype=dtype)
                
                for z, f in z_files:
                    if z < Z_SLICES:
                        img = skimage.io.imread(os.path.join(DATA_PATH, f))
                        vol[z] = img
                        
                # Write
                tifffile.imwrite(out_path, vol)
                
            except Exception as e:
                print(f"Failed to stack {{out_name}}: {{e}}")

    print("Stacking Complete.")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(output_dir, "stack_tiles.py"), "w") as f:
        f.write(script_content)

def generate_ometiff_converter(manifest: DatasetManifest, output_dir: str):
    """
    Generates convert_to_ometiff.py script for post-processing.
    Converts the raw binary output from nr_stitcher into OME-TIFF.
    """
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert NRStitcher raw output to OME-TIFF.
Generated by the NRStitcher Run Bundle Generator.
"""
import os
import sys
import glob
import numpy as np
import tifffile

# Dataset metadata
SAMPLE_NAME = "{manifest.dataset_name}"
VOXEL_X = {manifest.voxel_size_x_um}
VOXEL_Y = {manifest.voxel_size_y_um}
VOXEL_Z = {manifest.voxel_size_z_um}
BIT_DEPTH = {manifest.bit_depth}

def find_raw_output():
    """Find the stitched .raw output file (largest valid .raw)."""
    candidates = []
    # Try pattern based on sample name first
    candidates.extend(glob.glob(f"{{SAMPLE_NAME}}*.raw"))
    # Fallback to all raw files if none found
    if not candidates:
        candidates.extend(glob.glob("*.raw"))
        
    if not candidates:
        return None
        
    # Filter out known debug artifacts
    valid_candidates = []
    for f in candidates:
        if "defpoints" in f or "refpoints" in f or "gof" in f:
            continue
        valid_candidates.append(f)
        
    # If we filtered everything, revert to original candidates (just in case)
    if not valid_candidates:
        valid_candidates = candidates

    # Sort by size (largest first) - the stitched volume is huge
    valid_candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    return valid_candidates[0]

def read_raw_file(raw_path):
    """
    Read a raw binary file. 
    nr_stitcher writes a companion .txt file with dimensions.
    """
    # Look for dimension info
    # nr_stitcher typically creates files like: sample_stitch_complete.raw
    # and a corresponding sample_stitch_complete.txt or .hdr with dims
    base = os.path.splitext(raw_path)[0]
    
    # Try to find dimension file
    dim_file = None
    for ext in [".txt", ".hdr", ".dim"]:
        candidate = base + ext
        if os.path.exists(candidate):
            dim_file = candidate
            break
    
    if dim_file:
        print(f"Found dimension file: {{dim_file}}")
        with open(dim_file, "r") as f:
            content = f.read().strip()
        # Parse dimensions - format varies but usually "x y z" or "x,y,z"
        import re
        nums = re.findall(r"\\d+", content)
        if len(nums) >= 3:
            dims = [int(n) for n in nums[:3]]
            print(f"Parsed dimensions: {{dims}}")
        else:
            print(f"Could not parse dimensions from {{dim_file}}")
            dims = None
    else:
        dims = None
    
    # Determine dtype from bit depth
    if BIT_DEPTH == 8:
        dtype = np.uint8
    elif BIT_DEPTH == 16:
        dtype = np.uint16
    elif BIT_DEPTH == 32:
        dtype = np.float32
    else:
        dtype = np.uint16  # Default
    
    # Read raw data
    data = np.fromfile(raw_path, dtype=dtype)
    print(f"Read {{len(data)}} voxels ({{data.nbytes / 1e9:.2f}} GB)")
    
    if dims:
        # Reshape (Z, Y, X) - nr_stitcher convention
        try:
            vol = data.reshape(dims[2], dims[1], dims[0])
            print(f"Reshaped to (Z={{dims[2]}}, Y={{dims[1]}}, X={{dims[0]}})")
            return vol
        except ValueError as e:
            print(f"Reshape failed: {{e}}")
            print("Trying alternative dimension ordering...")
            try:
                vol = data.reshape(dims[0], dims[1], dims[2])
                print(f"Reshaped to {{vol.shape}}")
                return vol
            except ValueError:
                pass
    
    # If no dimension file or reshape failed, try to infer
    total = len(data)
    # Assume roughly cubic
    side = int(round(total ** (1/3)))
    print(f"No dimension info. Total voxels: {{total}}. Attempting cubic ({{side}}^3)...")
    
    # This is a fallback - unlikely to be correct
    print("WARNING: Could not determine dimensions. Please check output.")
    return data

def main():
    print("=" * 60)
    print("OME-TIFF Converter")
    print("=" * 60)
    
    raw_path = find_raw_output()
    if not raw_path:
        print("ERROR: No .raw output file found!")
        print("Make sure the stitcher has completed successfully.")
        sys.exit(1)
    
    print(f"Found raw output: {{raw_path}}")
    print(f"File size: {{os.path.getsize(raw_path) / 1e9:.2f}} GB")
    
    vol = read_raw_file(raw_path)
    
    if vol.ndim != 3:
        print(f"WARNING: Expected 3D volume, got {{vol.ndim}}D array of shape {{vol.shape}}")
        print("Skipping OME-TIFF conversion.")
        sys.exit(1)
    
    # Output path
    out_name = f"{{SAMPLE_NAME}}_stitched.ome.tif"
    print(f"Writing OME-TIFF: {{out_name}}")
    print(f"Volume shape: {{vol.shape}} (Z, Y, X)")
    
    # OME-TIFF metadata
    metadata = {{
        "axes": "ZYX",
        "PhysicalSizeX": VOXEL_X,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": VOXEL_Y,
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": VOXEL_Z,
        "PhysicalSizeZUnit": "µm",
    }}
    
    # Write with tifffile - bigtiff for large files, tile for performance
    tifffile.imwrite(
        out_name,
        vol,
        bigtiff=True,
        tile=(256, 256),
        compression="zlib",
        metadata=metadata,
        ome=True,
    )
    
    out_size = os.path.getsize(out_name) / 1e9
    print(f"Done! Output: {{out_name}} ({{out_size:.2f}} GB)")
    print("Open with Fiji, Napari, or QuPath.")

if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(output_dir, "convert_to_ometiff.py"), "w") as f:
        f.write(script_content)
