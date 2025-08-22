Write-Host "Installing Image Segmentation Project Dependencies in D Drive..." -ForegroundColor Green
Write-Host ""

# Change to D drive
Set-Location D:\

# Create virtual environment in D drive
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv "D:\Image Segmentation OpenCV\venv"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "D:\Image Segmentation OpenCV\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r "D:\Image Segmentation OpenCV\requirements.txt"

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: D:\Image Segmentation OpenCV\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To run the project: python main.py" -ForegroundColor Cyan

Read-Host "Press Enter to continue"
