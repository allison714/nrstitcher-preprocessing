from enum import Enum
import re
from typing import Tuple, Optional, List, Dict
import os
from dataclasses import dataclass
import numpy as np

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

def generate_stitch_settings(manifest: DatasetManifest, output_dir: str, data_path: str, use_tiles_view: bool, stitch_output_format: str = "raw", allow_warping: bool = True, binning: int = 1):
    """
    Generates stitch_settings.txt.
    If 'allow_warping' is True (Full Mode), it also generates 'stitch_settings_rigid_preview.txt' 
    for a fast initial check.
    """
    
    def _create_content(name_suffix: str, warp: bool, current_bin: int):
        # Calculate step sizes in pixels
        overlap_x_frac = manifest.overlap_x / 100.0
        overlap_y_frac = manifest.overlap_y / 100.0
        step_x = manifest.width_px * (1 - overlap_x_frac)
        step_y = manifest.height_px * (1 - overlap_y_frac)
        
        n_tiles = manifest.n_tiles_x * manifest.n_tiles_y
        lines = []
        lines.append(f"; Stitch Settings for {manifest.dataset_name}{name_suffix}")
        lines.append(f"; Generated by Antigravity Run Bundle Generator")
        lines.append("")
        lines.append("[stitch]")
        lines.append(f"sample_name = {manifest.dataset_name}{name_suffix}")
        lines.append(f"binning = {current_bin}")
        lines.append("point_spacing = 20")
        lines.append("coarse_block_radius = [25, 25, 25]")
        lines.append(f"coarse_binning = {max(4, current_bin * 2)}")
        lines.append("fine_block_radius = [25, 25, 25]")
        lines.append(f"fine_binning = {current_bin}")
        lines.append("normalize_in_blockmatch = True")
        lines.append("normalize_while_stitching = True")
        lines.append("global_optimization = True")
        lines.append("allow_rotation = True")
        lines.append(f"allow_local_deformations = {warp}")
        lines.append("zeroes_are_missing_values = True")
        lines.append(f"output_format = {stitch_output_format}")
        lines.append("")
        lines.append("[positions]")
        
        for t in range(n_tiles):
            grid_x, grid_y = tile_idx_to_xy(t, manifest.n_tiles_x, manifest.n_tiles_y, manifest.scan_order)
            pos_x = grid_x * step_x
            pos_y = grid_y * step_y
            pos_z = 0
            filename = f"stacks/tile_{t:03d}_ch0.tif" if manifest.n_channels > 1 else f"stacks/tile_{t:03d}.tif"
            
            def fmt(val):
                if isinstance(val, int): return str(val)
                return f"{val:.0f}" if val.is_integer() else f"{val:.1f}"
                
            lines.append(f"{filename} = {fmt(pos_x)}, {fmt(pos_y)}, {fmt(pos_z)}")
        return "\n".join(lines)

    # Main Settings
    main_content = _create_content("", allow_warping, binning)
    with open(os.path.join(output_dir, "stitch_settings.txt"), 'w') as f:
        f.write(main_content)
        
    # Optional Rigid Preview Settings (only if we are in Full Quality mode)
    if allow_warping:
        preview_content = _create_content("_rigid_preview", False, 2) # Rigid, 2x binned
        with open(os.path.join(output_dir, "stitch_settings_rigid_preview.txt"), 'w') as f:
            f.write(preview_content)

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


def generate_local_script(manifest: DatasetManifest, output_dir: str, conda_config: Dict[str, str], embed_pi2_path: Optional[str] = None, convert_neuroglancer: bool = False):
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

echo [INFO] Installing/Updating dependencies...
pip install numpy tifffile scikit-image networkx pyquaternion scipy tensorstore

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

REM --- PHASE 1: Fast Rigid Preview (if exists) ---
if exist "stitch_settings_rigid_preview.txt" (
    echo [INFO] Running Fast Rigid Preview...
    echo Command: {cmd_win} stitch_settings_rigid_preview.txt
    {cmd_win} stitch_settings_rigid_preview.txt
    if errorlevel 1 (
        echo [WARN] Rigid Preview failed. Continuing to full stitch...
    ) else (
        echo [SUCCESS] Rigid Preview complete.
    )
    echo.
)

REM --- PHASE 2: Main Stitch ---
echo [INFO] Running Main Stitch...
echo Command: {cmd_win} {stitch_args}
{cmd_win} {stitch_args}
if errorlevel 1 (
    echo [ERROR] Stitching failed!
    pause
    exit /b 1
)

{('echo.' + chr(10) + 'echo ============================================================' + chr(10) + 'echo   CONVERTING: Raw output to Neuroglancer Precomputed' + chr(10) + 'echo ============================================================' + chr(10) + 'python convert_to_neuroglancer.py' + chr(10) + 'if errorlevel 1 echo WARNING: Neuroglancer conversion failed. Raw output should still exist.') if convert_neuroglancer else ''}

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
        high = np.percentile(img_f, 99.9)
        
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
    echo "[INFO] Installing dependencies..."
    pip install numpy tifffile scikit-image networkx pyquaternion scipy tensorstore
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

# --- PHASE 1: Fast Rigid Preview (if exists) ---
if [ -f "stitch_settings_rigid_preview.txt" ]; then
    echo "[INFO] Running Fast Rigid Preview..."
    echo "Command: $STITCH_CMD stitch_settings_rigid_preview.txt"
    $STITCH_CMD stitch_settings_rigid_preview.txt
    if [ $? -ne 0 ]; then
        echo "[WARN] Rigid Preview failed. Continuing to full stitch..."
    else
        echo "[SUCCESS] Rigid Preview complete."
    fi
    echo ""
fi

# --- PHASE 2: Main Stitch ---
echo "[INFO] Running Main Stitch..."
echo "Command: $STITCH_CMD {stitch_args}"
$STITCH_CMD {stitch_args}
if [ $? -ne 0 ]; then
    echo "[ERROR] Stitching failed!"
    exit 1
fi

{"echo" + chr(10) + 'echo "============================================================"' + chr(10) + 'echo "  CONVERTING: Raw output to Neuroglancer Precomputed"' + chr(10) + 'echo "============================================================"' + chr(10) + "python convert_to_neuroglancer.py" + chr(10) + 'if [ $? -ne 0 ]; then' + chr(10) + '    echo "WARNING: Neuroglancer conversion failed. Raw output should still exist."' + chr(10) + "fi" if convert_neuroglancer else ""}

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


def generate_slurm_script(manifest: DatasetManifest, slurm_params: Dict, output_dir: str, conda_config: Dict[str, str], convert_neuroglancer: bool = False):
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

{'''
echo ""
echo "============================================================"
echo "  CONVERTING: Raw output to Neuroglancer Precomputed"
echo "============================================================"
# Attempt to install tensorstore if missing (cluster env might not have it)
pip install tensorstore --user -q
python convert_to_neuroglancer.py
''' if convert_neuroglancer else ""}

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
    [TABLED] Generates convert_to_ometiff.py script for post-processing.
    Converts the raw binary output from nr_stitcher into OME-TIFF.
    """
    # [Tabled in favor of Neuroglancer Precomputed]
    pass

def generate_neuroglancer_converter(manifest: DatasetManifest, output_dir: str, binning: int = 1):
    """
    Generates convert_to_neuroglancer.py script for post-processing.
    Converts the raw binary output from nr_stitcher into Neuroglancer Precomputed format using TensorStore.
    """
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert NRStitcher raw output to Neuroglancer Precomputed.
Generated by the NRStitcher Run Bundle Generator.
"""
import os
import sys
import glob
import numpy as np
import json

try:
    import tensorstore as ts
except ImportError:
    print("ERROR: 'tensorstore' package not found.")
    print("Please install it with: pip install tensorstore")
    sys.exit(1)

# Dataset metadata
SAMPLE_NAME = "{manifest.dataset_name}"
VOXEL_X_NM = {int(manifest.voxel_size_x_um * 1000)}
VOXEL_Y_NM = {int(manifest.voxel_size_y_um * 1000)}
VOXEL_Z_NM = {int(manifest.voxel_size_z_um * 1000)}
BIT_DEPTH = {manifest.bit_depth}
WIDTH = {manifest.width_px}
HEIGHT = {manifest.height_px}
# Actual dimensions will be parsed from the .txt/.hdr file

def find_raw_output():
    """Find the stitched .raw output file matching _<X>x<Y>x<Z>.raw and exclude trace files."""
    import re
    candidates = []
    
    # We look for all .raw files in the directory
    for f in glob.glob("*.raw"):
        # Ensure it has the _NxNxN size suffix
        if re.search(r"_\d+x\d+x\d+\.raw$", f):
            candidates.append(f)
            
    if not candidates:
        return None
        
    # Exclude trace artifacts
    valid_candidates = [f for f in candidates if not any(x in f for x in ["defpoints", "refpoints", "gof", "shifts"])]
    if not valid_candidates:
        valid_candidates = candidates

    valid_candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return valid_candidates[0]

def get_dimensions(raw_path):
    """Parse dimensions from companion .txt file or filename."""
    import re
    # 1. Try to parse from filename: e.g. _7065x7077x110.raw
    match = re.search(r"_(\d+)x(\d+)x(\d+)\.raw$", raw_path)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
        
    # 2. Try to parse from companion .txt
    base = os.path.splitext(raw_path)[0]
    for ext in [".txt", ".hdr", ".dim"]:
        candidate = base + ext
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                content = f.read()
            nums = re.findall(r"\d+", content)
            if len(nums) >= 3:
                return [int(n) for n in nums[:3]] # X, Y, Z
    
    # Fallback: Approximate from manifest settings if rigid preview or no txt
    print("WARNING: Could not parse exact dimensions from .txt/.hdr file. Using manifest estimates.")
    overlap_x_frac = {manifest.overlap_x} / 100.0
    overlap_y_frac = {manifest.overlap_y} / 100.0
    
    # Very rough estimate if it's a rigid preview
    est_w = int({manifest.width_px} + ({manifest.width_px} * (1 - overlap_x_frac)) * ({manifest.n_tiles_x} - 1))
    est_h = int({manifest.height_px} + ({manifest.height_px} * (1 - overlap_y_frac)) * ({manifest.n_tiles_y} - 1))
    est_d = {manifest.z_slices}
    
    # Adjust for binning
    is_preview = "rigid_preview" in raw_path
    binning = 2 if is_preview else {binning}
    
    return [max(1, est_w // binning), max(1, est_h // binning), est_d]

def main():
    print("=" * 60)
    print("Neuroglancer Precomputed Converter (TensorStore)")
    print("=" * 60)
    
    raw_path = find_raw_output()
    if not raw_path:
        print("ERROR: No .raw output file found!")
        sys.exit(1)
    
    dims = get_dimensions(raw_path)
    if not dims:
        print("ERROR: Could not determine dimensions from .txt or .hdr file.")
        sys.exit(1)
    
    # nr_stitcher: X, Y, Z. TensorStore/Neuroglancer: Z, Y, X or X, Y, Z depending on spec.
    # We'll use [X, Y, Z] for the underlying raw and map it.
    nx, ny, nz = dims
    print(f"Detected Dimensions: X={{nx}}, Y={{ny}}, Z={{nz}}")
    
    dtype = 'uint8' if BIT_DEPTH == 8 else 'uint16'
    
    # Define Neuroglancer Precomputed target
    output_path = "precomputed"
    target_spec = {{
        'driver': 'neuroglancer_precomputed',
        'kvstore': {{
            'driver': 'file',
            'path': output_path,
        }},
        'multiscale_metadata': {{
            'type': 'image',
            'data_type': dtype,
            'num_channels': 1,
        }},
        'scale_metadata': {{
            'size': [nx, ny, nz],
            'encoding': 'raw', # or 'compressed_segmentation' / 'jpeg'
            'chunk_size': [128, 128, 128],
            'resolution': [VOXEL_X_NM, VOXEL_Y_NM, VOXEL_Z_NM],
        }},
    }}

    print(f"Creating/Opening target precomputed: {{output_path}}")
    dataset = ts.open(target_spec, create=True, delete_existing=True).result()
    
    print(f"Opening source raw via memory map: {{raw_path}}")
    # pi2 writes flat binary in Z, Y, X order (C-contiguous)
    vol_zyx = np.memmap(raw_path, dtype=dtype, mode='r', shape=(nz, ny, nx))
    
    # TensorStore neuroglancer_precomputed expects (x, y, z, channel)
    vol_xyz = vol_zyx.transpose(2, 1, 0)
    vol_xyzc = np.expand_dims(vol_xyz, axis=3)
    
    print("Writing data in chunks (this may take a while)...")
    # TensorStore reads from the memmap chunks sequentially
    dataset.write(vol_xyzc).result()
    
    print(f"Success! Neuroglancer Precomputed output at: {{os.path.abspath(output_path)}}")
    print("You can view this by serving the folder with a CORS-enabled web server.")

if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(output_dir, "convert_to_neuroglancer.py"), "w") as f:
        f.write(script_content)

    # -------------------------------------------------------------
    # 2. Generate CORS Server Script (serve.py)
    # -------------------------------------------------------------
    serve_content = """#!/usr/bin/env python3
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a local CORS-enabled web server.')
    parser.add_argument('--port', type=int, default=8000, help='Port to serve on (default: 8000)')
    args = parser.parse_args()
    
    print(f"\\n=========================================")
    print(f" Serving Neuroglancer Data on port {args.port}")
    print(f"=========================================")
    print(f"1. Open Chrome/Firefox")
    print(f"2. Go to: https://neuroglancer-demo.appspot.com/")
    print(f"3. Add a new layer with Source: precomputed://http://localhost:{args.port}/precomputed")
    print(f"\\nPress Ctrl+C to stop the server.")
    
    try:
        HTTPServer(('localhost', args.port), CORSRequestHandler).serve_forever()
    except KeyboardInterrupt:
        print("\\nServer stopped.")
"""
    with open(os.path.join(output_dir, "serve.py"), "w") as f:
        f.write(serve_content)


def parse_tiff_stack(path: str) -> np.ndarray:
    """Helper to read a multi-page TIFF stack into a 3D numpy array (Z, Y, X)."""
    import tifffile
    return tifffile.imread(path)

# =============================================================================
# INTENSITY DRIFT ANALYSIS (NEW)
# =============================================================================
def analyze_intensity_drift(manifest_path: str, stacks_dir: str) -> Optional[dict]:
    """
    Analyzes the 3D stacks in the bundle to detect time/acquisition-dependent
    intensity drift. Uses a stratified subsampling approach for speed.
    """
    if not os.path.exists(manifest_path) or not os.path.exists(stacks_dir):
        return None

    import json
    import glob
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Reconstruct acquisition list from manifest metadata
    # We assume 'files' list aligns with acquisition order
    if 'files' not in manifest:
        return None
        
    tiles_data = []
    
    # We need to map original file index to the stack file (which is named bin{X}_tile_{%03d}tif...)
    # We will just parse the stacks_dir and extract indices, matching against manifest
    stack_files = glob.glob(os.path.join(stacks_dir, "*.raw")) + glob.glob(os.path.join(stacks_dir, "*.tif"))
    
    if not stack_files:
        return None

    import re
    
    # Stratified Subsampling Params
    NUM_Z_PLANES = 12
    CROP_PERCENT = 0.70
    
    for stack_path in stack_files:
        filename = os.path.basename(stack_path)
        # Extract the tile index assuming format ...tile_000.tif_... or similar
        match = re.search(r'tile_(\d+)', filename)
        if not match:
            continue
            
        acq_index = int(match.group(1))
        
        # Try to find position to determine column
        # Fallback grid assumption if positions fail
        col_index = acq_index  # Default fallback
        if "positions" in manifest and acq_index < len(manifest["positions"]):
            pos = manifest["positions"][acq_index]
            # Assumes Y is sweeping faster (column) or vice versa.
            # Usually, X determines the "column"
            col_index = int(pos['x']) # Simplified heuristic based on coordinate value
            
        try:
            if stack_path.endswith('.raw'):
                # We need dimensions to safely read .raw
                # Extract from filename e.g. _1600x1600x100.raw
                dim_match = re.search(r'_(\d+)x(\d+)x(\d+)\.raw$', filename)
                if dim_match:
                    nx, ny, nz = int(dim_match.group(1)), int(dim_match.group(2)), int(dim_match.group(3))
                    # Check dtype based on settings if possible, assume uint16 for stacked tiles
                    vol = np.memmap(stack_path, dtype=np.uint16, mode='r', shape=(nz, ny, nx))
                else:
                    continue # Cannot read raw without dims
            else:
                vol = parse_tiff_stack(stack_path) # Assume stacked TIF

            nz, ny, nx = vol.shape
            
            # Stratified Subsampling
            z_indices = np.linspace(0, nz - 1, NUM_Z_PLANES, dtype=int)
            
            y_start = int(ny * ((1 - CROP_PERCENT) / 2))
            y_end = int(y_start + ny * CROP_PERCENT)
            x_start = int(nx * ((1 - CROP_PERCENT) / 2))
            x_end = int(x_start + nx * CROP_PERCENT)
            
            # Extract planar slices
            slices = vol[z_indices, y_start:y_end, x_start:x_end]
            
            # Subsample XY per plane (e.g. step by 4) to hit ~200k voxels total
            step = 4
            subsampled = slices[:, ::step, ::step].flatten()
            
            if len(subsampled) > 0:
                p90 = float(np.percentile(subsampled, 90))
                p50 = float(np.percentile(subsampled, 50))
                p10 = float(np.percentile(subsampled, 10))
                
                tiles_data.append({
                    'acq_index': acq_index,
                    'col_index': col_index,
                    'p90': p90,
                    'p50': p50,
                    'p10': p10,
                    'filename': filename,
                    'path': stack_path,
                    'shape': (nz, ny, nx)
                })
                
        except Exception as e:
            print(f"Failed to analyze {filename}: {e}")
            continue

    if not tiles_data:
        return None
        
    # Sort by acquisition index
    tiles_data.sort(key=lambda x: x['acq_index'])
    
    # In a true column-serpentine scan, the 'col_index' might be difficult to guess dynamically 
    # without exact stage coordinates. We group strictly by unique X-positions if possible,
    # otherwise we default to the acq_index itself (treating every tile independently).
    
    # Calculate global metrics
    p90_array = np.array([t['p90'] for t in tiles_data])
    acq_array = np.array([t['acq_index'] for t in tiles_data])
    
    # Calculate correlation (using numpy to avoid scipy dependency issues)
    corr = 0.0
    if len(p90_array) > 1 and np.std(p90_array) > 0 and np.std(acq_array) > 0:
        corr = float(np.corrcoef(acq_array, p90_array)[0, 1])
    
    pct_drop = 0
    if len(p90_array) > 1 and p90_array[0] > 0:
        pct_drop = ((p90_array[0] - p90_array[-1]) / p90_array[0]) * 100
        
    return {
        'tiles': tiles_data,
        'correlation': corr,
        'percent_drop': pct_drop
    }

def generate_gain_corrected_stacks(manifest_path: str, drift_data: dict, source_stacks_dir: str, target_stacks_dir: str):
    """
    Reads the original stacked volumes, applies a gain normalization to counter intensity drift,
    and writes them out to a new folder alongside a new `stitch_settings.txt` that points to them.
    """
    if not os.path.exists(target_stacks_dir):
        os.makedirs(target_stacks_dir)
        
    tiles = drift_data['tiles']
    
    # We will normalize every tile so that its p90 matches the global maximum p90.
    # We use a column median to smooth it out if they are grouped, otherwise smooth over local neighborhood
    
    p90_max = max(t['p90'] for t in tiles)
    
    withst = open(os.path.join(os.path.dirname(target_stacks_dir), "gain_corrected_stitch_settings.txt"), "w")
    
    for t in tiles:
        # Calculate local target gain
        # If we have distinct columns, we'd use the column median. 
        # For a generic robust approach, we just use the tile's own p90 relative to max,
        # bounded to prevent insane boosting (e.g. max gain of 3.0x)
        
        target_gain = p90_max / t['p90'] if t['p90'] > 0 else 1.0
        target_gain = min(target_gain, 3.0) 
        
        source_path = t['path']
        filename = t['filename']
        target_path = os.path.join(target_stacks_dir, filename)
        
        nz, ny, nx = t['shape']
        
        # Read source
        if source_path.endswith('.raw'):
            vol = np.memmap(source_path, dtype=np.uint16, mode='r', shape=(nz, ny, nx))
        else:
            vol = parse_tiff_stack(source_path)
            
        # Apply gain in float32, then clip and convert back to uint16
        print(f"Applying gain {target_gain:.2f}x to {filename}...")
        
        # Write to target (using physical rewrite for full compatibility)
        # Chunk the conversion to avoid RAM blowing up
        target_vol = np.memmap(target_path, dtype=np.uint16, mode='w+', shape=(nz, ny, nx))
        
        chunk_z = 10
        for z in range(0, nz, chunk_z):
            z_end = min(nz, z + chunk_z)
            chunk = vol[z:z_end].astype(np.float32)
            chunk = chunk * target_gain
            np.clip(chunk, 0, 65535, out=chunk)
            target_vol[z:z_end] = chunk.astype(np.uint16)
            
        target_vol.flush()
        
        # Write out the entry for the new stitch_settings.txt
        # We need the original manifest position block
        # For simplicity, we assume the user will just replace the `[positions]` block in their main file,
        # but here we generate a complete minimal valid positional string.
        file_path_for_pi2 = os.path.join("gain_corrected_stacks", filename).replace("\\", "/")
        withst.write(f"file: {file_path_for_pi2}\\n")
        
    withst.close()


def parse_alignment_points(file_path: str) -> Optional[np.ndarray]:
    """
    Parses alignment point files.
    - If .txt: Assumes N lines, each having 3 coordinates (X, Y, Z).
    - If .raw: Assumes float32 flat array of vectors (typically displacement).
    Returns a numpy array of shape (N, 3) or None if parsing fails.
    """
    if not os.path.exists(file_path):
        return None
        
    if file_path.endswith('.raw'):
        try:
            data = np.fromfile(file_path, dtype=np.float32)
            if len(data) > 0 and len(data) % 3 == 0:
                return data.reshape(-1, 3)
            return None
        except:
            return None
    
    try:
        # Try reading with numpy
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        return data
    except Exception as e:
        # Fallback manual parsing in case of complex headers/footers
        try:
            points = []
            with open(file_path, 'r') as f:
                for line in f:
                    # Extract numbers
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if len(nums) >= 3:
                        points.append([float(n) for n in nums[:3]])
            return np.array(points) if points else None
        except:
            return None


def parse_local_shift_files(bundle_dir: str, subsample_step: int = 10, overlap_margin: float = 0.15) -> Optional[np.ndarray]:
    """
    Parses and aggregates all high-resolution `world_to_local_shifts` files in a bundle directory.
    Downsamples the 3D grid and returns a 1D array of displacement magnitudes to prevent memory overload.
    
    Args:
        bundle_dir: Path to the run bundle directory (often the 'trace' subfolder or root).
        subsample_step: Voxel jump step to safely reduce millions of points to thousands.
        overlap_margin: Fraction of the tile edge to keep (e.g. 0.15 for 15%). If 0, keeps entire tile.
                        This strictly focuses the analysis on the overlap peripheries where pi2
                        algorithms do their non-linear matching, rejecting interpolated rigid centers.
        
    Returns:
        1D numpy array of absolute vector magnitudes (in pixels), or None if no files exist.
    """
    import glob
    import re
    
    # Check trace folder first, then bundle root
    candidates = []
    for search_dir in [os.path.join(bundle_dir, "trace"), bundle_dir]:
        if os.path.exists(search_dir):
            # Glob for shift files natively output by recent pi2 versions
            # They can look like: bin2_tile_000tif_world_to_local_shifts_162x162x12.raw
            # or: stacks-tile_000tif_world_to_local_shifts_322x322x22.raw
            candidates.extend(glob.glob(os.path.join(search_dir, "*world_to_local_shifts_*.raw")))
            
    if not candidates:
        return None
        
    all_magnitudes = []
    all_x = []
    all_y = []
    all_dx = []
    all_dy = []
    all_dz = []
    worst_tiles = []
    max_tile_size = 0
    
    # Try to extract fine_binning or binning from stitch_settings.txt
    binning_val = 1.0
    settings_path = os.path.join(bundle_dir, "stitch_settings.txt")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as sf:
                content = sf.read()
                # Look for fine_binning or binning=
                match_fine = re.search(r'fine_binning\s*=\s*([\d\.]+)', content)
                match_bin = re.search(r'binning\s*=\s*([\d\.]+)', content)
                if match_fine:
                    binning_val = float(match_fine.group(1))
                elif match_bin:
                    binning_val = float(match_bin.group(1))
        except Exception:
            pass
            
    # Also check if it's explicitly named bin2, bin4 in the filename prefix
    if binning_val == 1.0 and candidates:
        if "bin2" in os.path.basename(candidates[0]): binning_val = 2.0
        elif "bin4" in os.path.basename(candidates[0]): binning_val = 4.0
    
    for f in candidates:
        # Extract dimensions from the filename (e.g., ..._162x162x12.raw)
        match = re.search(r'_(\d+)x(\d+)x(\d+)\.raw$', f)
        if not match:
            continue
            
        nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))
        expected_size = nx * ny * nz * 3
        max_tile_size = max(max_tile_size, max(nx, ny))
        
        try:
            # Read flat 32-bit float array
            data = np.fromfile(f, dtype=np.float32)
            
            # Basic sanity check
            if len(data) != expected_size:
                # If size mismatches exactly, we skip this outlier
                continue
                
            # Reshape into X, Y, Z, 3 spatial map
            data_3d = data.reshape((nx, ny, nz, 3))
            
            # Generate 3D grid of coordinates to track vector origins
            coords_x, coords_y, _ = np.mgrid[0:nx, 0:ny, 0:nz]
            
            # Normalize X and Y to [0.0, 1.0] representing generic tile spatial area
            x_norm = coords_x / float(nx)
            y_norm = coords_y / float(ny)
            
            if overlap_margin > 0:
                edge_x = max(1, int(nx * overlap_margin))
                edge_y = max(1, int(ny * overlap_margin))
                
                # Create a boolean mask indicating the overlapping edge regions
                mask = np.zeros((nx, ny, nz), dtype=bool)
                mask[:edge_x, :, :] = True      # Left overlap
                mask[-edge_x:, :, :] = True     # Right overlap
                mask[:, :edge_y, :] = True      # Bottom overlap
                mask[:, -edge_y:, :] = True     # Top overlap
                
                # Subsample data and mask identically to save memory
                sampled_data = data_3d[::subsample_step, ::subsample_step, ::subsample_step, :]
                sampled_mask = mask[::subsample_step, ::subsample_step, ::subsample_step]
                
                sampled_x = x_norm[::subsample_step, ::subsample_step, ::subsample_step]
                sampled_y = y_norm[::subsample_step, ::subsample_step, ::subsample_step]
                
                # Extract only the vectors inside the bounding overlap regions
                sampled_vectors = sampled_data[sampled_mask]
                filtered_x = sampled_x[sampled_mask]
                filtered_y = sampled_y[sampled_mask]
            else:
                sampled_data = data_3d[::subsample_step, ::subsample_step, ::subsample_step, :]
                sampled_vectors = sampled_data.reshape(-1, 3)
                filtered_x = x_norm[::subsample_step, ::subsample_step, ::subsample_step].flatten()
                filtered_y = y_norm[::subsample_step, ::subsample_step, ::subsample_step].flatten()
            
            # Subtract the global baseline translation (median shift of the tile)
            # This isolates the true relative (non-linear) deformation vectors dx, dy, dz
            baseline_shift = np.median(sampled_vectors, axis=0)
            warping_vectors = sampled_vectors - baseline_shift
            
            # Calculate magnitude (vector length) of true non-rigid displacement
            magnitudes = np.linalg.norm(warping_vectors, axis=-1)
            
            local_max = np.max(magnitudes) if len(magnitudes) > 0 else 0
            worst_tiles.append((local_max, os.path.basename(f)))
            
            # Collect individual components
            all_dx.append(warping_vectors[:, 0])
            all_dy.append(warping_vectors[:, 1])
            all_dz.append(warping_vectors[:, 2])
            
            # Accumulate into our master list
            all_magnitudes.append(magnitudes)
            all_x.append(filtered_x)
            all_y.append(filtered_y)
        except Exception as e:
            print(f"Warning: Failed to parse {f}: {e}")
            continue
            
    if all_magnitudes:
        worst_tiles.sort(key=lambda x: x[0], reverse=True)
        # Concatenate all tile magnitudes into one big statistical pool
        return {
            'magnitudes': np.concatenate(all_magnitudes),
            'dx': np.concatenate(all_dx),
            'dy': np.concatenate(all_dy),
            'dz': np.concatenate(all_dz),
            'x': np.concatenate(all_x),
            'y': np.concatenate(all_y),
            'worst_tiles': worst_tiles,
            'max_tile_size': max_tile_size,
            'overlap_margin': overlap_margin,
            'binning_scale': binning_val
        }
        
    return None
