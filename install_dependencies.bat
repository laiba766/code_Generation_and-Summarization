@echo off
echo Installing Python dependencies for Phase 3...
echo.

REM Install packages one by one to show progress
echo [1/20] Installing numpy...
C:\Python313\python.exe -m pip install numpy>=1.24.0 --quiet

echo [2/20] Installing pandas...
C:\Python313\python.exe -m pip install pandas>=2.0.0 --quiet

echo [3/20] Installing scikit-learn...
C:\Python313\python.exe -m pip install scikit-learn>=1.3.0 --quiet

echo [4/20] Installing matplotlib...
C:\Python313\python.exe -m pip install matplotlib>=3.7.0 --quiet

echo [5/20] Installing seaborn...
C:\Python313\python.exe -m pip install seaborn>=0.12.0 --quiet

echo [6/20] Installing PyYAML...
C:\Python313\python.exe -m pip install pyyaml>=6.0 --quiet

echo [7/20] Installing tqdm...
C:\Python313\python.exe -m pip install tqdm>=4.65.0 --quiet

echo [8/20] Installing requests...
C:\Python313\python.exe -m pip install requests>=2.31.0 --quiet

echo [9/20] Installing jsonlines...
C:\Python313\python.exe -m pip install jsonlines>=3.1.0 --quiet

echo [10/20] Installing radon...
C:\Python313\python.exe -m pip install radon>=6.0.1 --quiet

echo [11/20] Installing plotly...
C:\Python313\python.exe -m pip install plotly>=5.14.0 --quiet

echo [12/20] Installing networkx...
C:\Python313\python.exe -m pip install networkx>=3.1 --quiet

echo [13/20] Installing torch...
C:\Python313\python.exe -m pip install torch>=2.0.0 --quiet

echo [14/20] Installing transformers...
C:\Python313\python.exe -m pip install transformers>=4.30.0 --quiet

echo [15/20] Installing umap-learn...
C:\Python313\python.exe -m pip install umap-learn>=0.5.3 --quiet

echo [16/20] Installing hdbscan...
C:\Python313\python.exe -m pip install hdbscan>=0.8.33 --quiet

echo [17/20] Installing lizard...
C:\Python313\python.exe -m pip install lizard>=1.17.10 --quiet

echo [18/20] Installing python-dotenv...
C:\Python313\python.exe -m pip install python-dotenv>=1.0.0 --quiet

echo [19/20] Installing astroid...
C:\Python313\python.exe -m pip install astroid>=2.15.0 --quiet

echo [20/20] Installing tree-sitter...
C:\Python313\python.exe -m pip install tree-sitter>=0.20.0 --quiet

echo.
echo ========================================
echo All dependencies installed successfully!
echo ========================================
echo.
echo You can now run: python main.py
echo.
pause
