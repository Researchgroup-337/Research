@echo off
cd /d "%~dp0\.."
set PYTHONUTF8=1
echo === Rebuilding all results with your trained DistilBERT (about 20-30 minutes) ===
python scripts\check_setup.py --rebuild || goto :fail
python scripts\prepare_data.py || goto :fail
python scripts\generate_synthetic.py || goto :fail
python scripts\fit_risk_model.py || goto :fail
python scripts\evaluate.py || goto :fail
echo.
echo DONE. Open reports\EVALUATION_REPORT.md
pause
exit /b 0
:fail
echo.
echo *** A step failed. Copy the red/error text above and send it to Claude. ***
pause
exit /b 1
