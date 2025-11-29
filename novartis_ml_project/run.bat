@echo off
REM Activate virtual environment and run main pipeline

echo ========================================
echo Novartis Datathon 2025 - ML Pipeline
echo ========================================
echo.

REM Check if Python is available (try 'python' first, then 'py')
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found!
        echo Please install Python 3.8+ and add it to PATH
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create it
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo.
echo Checking dependencies...
python -c "import lightgbm, xgboost" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: Dependencies not installed!
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements!
        pause
        exit /b 1
    )
)

REM Run main pipeline
echo.
echo ========================================
echo Starting ML Pipeline...
echo ========================================
echo.

python main.py

REM Check if script completed successfully
if errorlevel 1 (
    echo.
    echo ERROR: Pipeline failed!
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo Pipeline completed successfully!
    echo ========================================
    echo.
    echo Check outputs/ for results
    echo Check logs/ for training logs
    echo.
)

pause

