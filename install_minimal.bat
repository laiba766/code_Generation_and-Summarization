@echo off
echo Minimal Install - Core packages only (for testing)
echo ==================================================
echo.

echo Installing core packages...
C:\Python313\python.exe -m pip install pyyaml numpy pandas scikit-learn matplotlib seaborn tqdm

echo.
echo ========================================
echo Core packages installed!
echo ========================================
echo.
echo Note: For full functionality, run install_quick.bat
echo.
pause
