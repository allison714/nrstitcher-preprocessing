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

### 2. Getting Started

There are two ways to start the application, depending on your operating system:

**Option A: Windows (Easiest)**
An automated batch script is provided that will automatically find your Conda installation, create the required `stitch_app` environment, install dependencies, and launch the UI.
1. Simply double-click **`RUN_ME.bat`** in the project folder!

**Option B: Mac / Linux (Manual Anaconda Prompt)**
If you are on Mac/Linux, or prefer to set up the environment manually:
1. Open your terminal (or **Anaconda Prompt**).
2. Navigate to the downloaded project directory: `cd path/to/nrstitcher-preprocessing`
3. Create and activate the conda environment, then install dependencies:
```bash
conda create -n stitch_app python=3.9 -y
conda activate stitch_app
pip install -r requirements.txt
```
4. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser automatically.

## Workflow

1.  **Select Data**: Input your dimensional parameters, overlapping preferences, and physical resolutions.
![Data Configuration](file:///C:/Users/allis/.gemini/antigravity/brain/a36c68f1-6232-4658-8d75-fabd2bf012b9/screenshot_inputs.png)

2.  **Verify**: Use the **"Preview Tiles"** section to visually check if your coordinate mappings follow your intended scan path.

3.  **Configure Execution**:
    *   **Target Environment**: Choose "Local Workstation" or "Misha Cluster (Slurm)".
![Execution Setup](file:///C:/Users/allis/.gemini/antigravity/brain/a36c68f1-6232-4658-8d75-fabd2bf012b9/screenshot_tooltip.png)
    *   **Backend config**: The app attempts to auto-detect `pi2` or `nrstitcher`. You can override this if needed.

4.  **Generate**: Click "Generate Run Bundle".

## Post-Generation Quality Control (QC)

After generating and running your stitched bundle, point the application to your output directory to compute advanced quality control analytics.

### Warping Diagnostics
Identifies mechanical issues by isolating non-linear 3D spatial drags across tile boundaries. Pinpoints exact regions of problematic physical topologies using robust median density filters.
![Warping Diagnostics](file:///C:/Users/allis/.gemini/antigravity/brain/a36c68f1-6232-4658-8d75-fabd2bf012b9/screenshot_warping.png)

### Intensity Drift Analysis
Mathematically decouples baseline illumination decay and laser-power inconsistencies from actual physiological density measurements. This allows for safe, robust detection of photobleaching trajectories and automatically scaffolds a physical structural file rewrite (Gain Correction) if unacceptable attenuation is observed.
![Drift Analysis](file:///C:/Users/allis/.gemini/antigravity/brain/a36c68f1-6232-4658-8d75-fabd2bf012b9/screenshot_drift.png)

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
1.  Navigate to the generated bundle folder.
2.  Double-click **`run_local.bat`** (Windows) or run `./run_local.sh` (Linux/Mac).
3.  The script will:
    *   Activate the `stitch_app` environment.
    *   Run the stitch command.
    *   (Optional) Convert the output to Neuroglancer Precomputed format if selected.

## Neuroglancer Visualization
If you selected **Neuroglancer Precomputed** output, you can instantly view your 3D stitched volume in your browser:

1. Open **Anaconda Prompt**.
2. Navigate to your stitched output folder (e.g., `cd D:\StitchScratch\260224_local...`).
3. Run the included server script:
   ```bash
   python serve.py
   ```
4. Open Chrome and go to [neuroglancer-demo.appspot.com](https://neuroglancer-demo.appspot.com/).
5. Click the `+` icon in the top left and add your local Source: `precomputed://http://localhost:8000/precomputed`.

## Output
The app creates a new folder containing:
*   `stitch_settings.txt`: Configuration file.
*   `dataset_manifest.json`: JSON record of settings.
*   `run_local.bat` / `run_local.sh`: Intelligent execution scripts.
*   `serve.py`: Local web server for Neuroglancer visualization.
*   `tools/`: (If using Portable Bundle) Contains the embedded `pi2` package.

## Troubleshooting

### Auto-Install Failed
*   **Error**: `pip install git+... failed`
*   **Cause**: You likely don't have `git` installed, or you are behind a firewall.
*   **Fix**: Download `pi2` manually from GitHub, extract it, and use the **"Path to 'pi2' Package Source"** field in the App to create a Portable Bundle instead.

### "Conda not found"
*   The script tries to find `conda` automatically. If it fails, open the App and check **"Local Config > Conda Init Script"**. Ensure it points to your actual `conda.bat` (usually `C:\Users\Username\anaconda3\condabin\conda.bat`).

## Exporting this README to PDF
If you use Visual Studio Code and wish to save this documentation as a PDF, you can automate it using the `Markdown PDF` extension:
1. Install the `Markdown PDF` extension by *yzane*.
2. Add the `"markdown-pdf.convertOnSave": true` option to your VS Code `settings.json`.
3. Restart Visual Studio Code.
4. Open this Markdown file and save it (Ctrl+S / Cmd+S). The PDF will auto-generate in the same folder.
