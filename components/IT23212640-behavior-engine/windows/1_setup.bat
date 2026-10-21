@echo off
cd /d "%~dp0\.."
set PYTHONUTF8=1
echo === Installing Python libraries (5-10 minutes) ===
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
echo.
echo === Checking the installation ===
python scripts\check_setup.py
pause
