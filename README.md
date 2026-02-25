# Run Bundle Generator for pi2/NRStitcher

This Streamlit application generates ready-to-execute "Run Bundles" for the [pi2/NRStitcher](https://github.com/abc/nrstitcher) pipeline. It simplifies the complex process of creating configuration files and execution scripts for both local workstations and Slurm-managed clusters (like Misha).

## Features

*   **Interactive Configuration**: Easily input dataset parameters (dimensions, overlap, voxel size) via a GUI.
*   **Auto-Detection**: Automatically detects your dataset dimensions from file metadata.
*   **Smart Script Generation**: Creates `run_local.bat` (Windows), `run_local.sh` (Linux/Mac), and `run_nrstitcher.sbatch` (Slurm) with intelligent backend detection.
*   **Tiles View**: (Optional) Creates a `tiles/` folder with symlinks, renaming your files to a structured format (`tile_{t}_z_{z}_c_{c}.tif`) expected by some viewers, without duplicating data.
*   **Preview**: Visually inspect your tiles and verify coordinate mapping before generating.
*   **Slurm Resource Recommendation**: Estimates required Partition, CPU, Memory, and Time based on your dataset size.

## Quick Start

### 1. Get the Code & Stitcher

You'll need this UI application and the core stitcher (`pi2`).

**A. The Run Bundle App (This UI)**
Clone this repository to your machine:
```bash
git clone https://github.com/allison714/nrstitcher-preprocessing.git
cd nrstitcher-preprocessing
```

**B. The Core Stitcher (`pi2`)**
*   **Windows (Recommended)**: Download the pre-compiled binary distribution (`pi2-v4.5-win-no-opencl`). The app automatically supports it if placed at `D:\pi2-v4.5-win-no-opencl` (or you can link it manually). You do NOT need to clone the pi2 repo.
*   **Source Building**: If you must build from source, clone the `pi2` repo (`git clone https://github.com/arttumiettinen/pi2.git`).

### 2. Installation

Ensure you have Python 3.9+ and `conda` installed.

```bash
# Create environment
conda create -n stitch_app python=3.9 -y
conda activate stitch_app

# Install dependencies
pip install streamlit tifffile pandas
```

### 2. Running the App

Navigate to the project directory and run:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Workflow

1.  **Select Data**: detailed instruction in the UI.
2.  **Verify**: Use the **"Preview Tiles"** section to check if your files are being read correctly.
3.  **Configure Execution**:
    *   **Target Environment**: Choose "Local Workstation" or "Misha Cluster (Slurm)".
    *   **Backend config**: The app attempts to auto-detect `pi2` or `nrstitcher`. You can override this if needed.
4.  **Generate**: Click "Generate Run Bundle".

## Output

The app creates a new folder (e.g., `MyDataset_local` or `MyDataset_slurm`) containing:

*   `stitch_settings.txt`: The coordinate configuration for the stitcher.
*   `dataset_manifest.json`: A record of your settings for reproducibility.
*   `run_local.bat` / `run_local.sh` / `run_nrstitcher.sbatch`: The specific script to **run the actual stitching**.
*   `tiles/`: (Optional) The symlinked view of your data.

## Local Workstation Execution

### Prerequisite: `pi2` / `nrstitcher`
For local stitching, you need the `pi2` software. Since this is not a public PyPI package, you have two options:

1.  **Portable Bundle (Recommended)**: 
    *   Download the [pi2 source code](https://github.com/arttumiettinen/pi2).
    *   In the App, under **Local Config**, paste the path to your `pi2` folder in **"Path to 'pi2' Package Source"**.
    *   The App will **embed** a copy of `pi2` into the run bundle (`tools/pi2`).
    *   The generated script will automatically use this embedded copy, meaning you don't need to install it in your environment!

2.  **Auto-Download from GitHub**:
    *   If you leave the "Path to 'pi2'" blank, the generated script will attempt to:
        1.  Create a fresh Conda environment (`stitch_app`) if it doesn't exist.
        2.  Install dependencies (`numpy`, `tifffile`, `scikit-image`).
        3.  Run `pip install git+https://github.com/arttumiettinen/pi2`.
    *   **Requires**: `git` must be installed and available in your command prompt.

### Running the Stitcher
1.  Navigate to the generated bundle folder (e.g., `MyDataset_local`).
2.  Double-click **`run_local.bat`** (Windows) or run `./run_local.sh` (Linux/Mac).
3.  The script will:
    *   Activate (or create) the `stitch_app` environment.
    *   Check for `pi2`.
    *   If missing, it will use the **embedded copy** or try to **auto-download** it.
    *   Run the stitch command.

## Output
The app creates a new folder containing:
*   `stitch_settings.txt`: Configuration file.
*   `dataset_manifest.json`: JSON record of settings.
*   `run_local.bat` / `run_local.sh`: Intelligent execution scripts.
*   `tools/`: (If using Portable Bundle) Contains the embedded `pi2` package.
*   `tiles/`: (Optional) Symlinked view of raw data.

## Troubleshooting

### Auto-Install Failed
*   **Error**: `pip install git+... failed`
*   **Cause**: You likely don't have `git` installed, or you are behind a firewall.
*   **Fix**: Download `pi2` manually from GitHub, extract it, and use the **"Path to 'pi2' Package Source"** field in the App to create a Portable Bundle instead.

### "Conda not found"
*   The script tries to find `conda` automatically. If it fails, open the App and check **"Local Config > Conda Init Script"**. Ensure it points to your actual `conda.bat` (usually `C:\Users\Username\anaconda3\condabin\conda.bat`).
