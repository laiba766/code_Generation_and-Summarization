@echo off
echo Quick Install - Essential packages only
echo ========================================
echo.

echo Installing all packages from requirements.txt...
C:\Python313\python.exe -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo You can now run: python main.py
echo.
pause
