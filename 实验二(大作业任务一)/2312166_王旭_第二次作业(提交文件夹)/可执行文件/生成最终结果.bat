@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD=python"
python --version >nul 2>nul
if errorlevel 1 (
    set "PYTHON_CMD=py -3"
    py -3 --version >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.8 or later and add it to PATH.
        pause
        exit /b 1
    )
)

set "PREDICTION_SCRIPT="
for /d %%D in ("..\*") do (
    if exist "%%~fD\generate_predictions.py" set "PREDICTION_SCRIPT=%%~fD\generate_predictions.py"
)

if not defined PREDICTION_SCRIPT (
    echo generate_predictions.py was not found in a sibling directory.
    echo Keep the source and executable directories at the same level.
    pause
    exit /b 1
)

echo Training the final ensemble and generating predictions. Please wait...
%PYTHON_CMD% "%PREDICTION_SCRIPT%" --train "data\train.txt" --test "data\test.txt" --output "output\final\result.txt"

if errorlevel 1 (
    echo Execution failed. Check the error message above.
) else (
    echo Completed. Result: output\final\result.txt
)
pause
endlocal
