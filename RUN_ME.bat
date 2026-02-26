@echo off
echo ===================================================
echo Slurm Run Bundle Generator - Auto Start Script
echo ===================================================

:: Ensure script always runs from its own containing directory
cd /d "%~dp0"

echo [INFO] Searching for Python/Conda installation...
set "CONDA_ACTIVATE="

:: Check common installation paths
for %%p in (
    "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    "%PROGRAMDATA%\anaconda3\Scripts\activate.bat"
    "%PROGRAMDATA%\miniconda3\Scripts\activate.bat"
    "C:\anaconda3\Scripts\activate.bat"
    "C:\miniconda3\Scripts\activate.bat"
    "C:\Program Files\anaconda3\Scripts\activate.bat"
) do (
    if exist %%p (
        set "CONDA_ACTIVATE=%%p"
        goto :conda_found
    )
)

:conda_not_found
echo.
echo [ERROR] Anaconda or Miniconda could not be found in standard locations.
echo ---------------------------------------------------
echo MISSING DOWNLOAD: Anaconda / Miniconda 3
echo ---------------------------------------------------
echo To fix this, please install Miniconda:
echo Download: https://docs.conda.io/en/latest/miniconda.html
echo Installation Tip: When installing, choose "Just Me" so it installs to your %%USERPROFILE%% folder.
echo.
pause
exit /b 1

:conda_found
echo [INFO] Found Conda environment script at: "%CONDA_ACTIVATE%"
call "%CONDA_ACTIVATE%"

echo [INFO] Checking for 'stitch_app' workspace environment...
call conda env list | findstr /i "stitch_app" >nul
if errorlevel 1 (
    echo [INFO] Environment 'stitch_app' not found. Creating it immediately (Python 3.9)...
    call conda create -n stitch_app python=3.9 -y
    call conda activate stitch_app
    
    echo [INFO] Installing required Python packages from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [INFO] Environment 'stitch_app' formally found. Activating...
    call conda activate stitch_app
)

echo.
echo [INFO] Starting the Streamlit Graphical Interface...
echo Please do not close this black window while using the application!
echo.
streamlit run app.py

echo.
echo [INFO] Streamlit session ended or encountered a terminal error.
pause
