@echo off
echo Installing Image Segmentation Project Dependencies in D Drive...
echo.

REM Change to D drive
D:

REM Create virtual environment in D drive
echo Creating virtual environment...
python -m venv "D:\Image Segmentation OpenCV\venv"

REM Activate virtual environment
echo Activating virtual environment...
call "D:\Image Segmentation OpenCV\venv\Scripts\activate.bat"

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r "D:\Image Segmentation OpenCV\requirements.txt"

echo.
echo Installation complete! 
echo To activate the environment, run: "D:\Image Segmentation OpenCV\venv\Scripts\activate.bat"
echo To run the project: python main.py
pause
