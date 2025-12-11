# Installation Instructions

## Quick Fix for Your Error

You got `ModuleNotFoundError: No module named 'yaml'` because the dependencies aren't installed yet.

## Installation Options

### Option 1: Quick Install (Recommended)

Double-click `install_quick.bat` or run in PowerShell:

```powershell
.\install_quick.bat
```

This will install all packages from `requirements.txt` at once.

### Option 2: Minimal Install (For Quick Testing)

If you just want to test if Python is working:

```powershell
.\install_minimal.bat
```

This installs only core packages (PyYAML, NumPy, Pandas, etc.)

### Option 3: Manual Installation (If batch files don't work)

Open PowerShell in the project folder and run:

```powershell
C:\Python313\python.exe -m pip install -r requirements.txt
```

Or install packages individually:

```powershell
# Core packages (required)
C:\Python313\python.exe -m pip install pyyaml
C:\Python313\python.exe -m pip install numpy pandas
C:\Python313\python.exe -m pip install scikit-learn
C:\Python313\python.exe -m pip install matplotlib seaborn

# ML packages (for experiments)
C:\Python313\python.exe -m pip install torch
C:\Python313\python.exe -m pip install transformers

# Additional utilities
C:\Python313\python.exe -m pip install tqdm requests jsonlines
C:\Python313\python.exe -m pip install radon plotly networkx
```

## Verify Installation

After installing, verify it worked:

```powershell
C:\Python313\python.exe -c "import yaml, numpy, pandas, sklearn; print('Success! All packages installed.')"
```

## Run the Project

Once installed, you can run:

```powershell
# Run full pipeline
C:\Python313\python.exe main.py --step all

# Or run specific steps
C:\Python313\python.exe main.py --step extract
C:\Python313\python.exe main.py --step cluster
C:\Python313\python.exe main.py --step predict
```

## Troubleshooting

### Issue: "pip is not recognized"

Solution:
```powershell
C:\Python313\python.exe -m ensurepip
C:\Python313\python.exe -m pip install --upgrade pip
```

### Issue: "Permission denied"

Solution: Run PowerShell as Administrator, or use:
```powershell
C:\Python313\python.exe -m pip install --user -r requirements.txt
```

### Issue: Installation is too slow

Solution: Use minimal install first, then install additional packages as needed:
```powershell
.\install_minimal.bat
```

### Issue: Torch installation fails

Solution: PyTorch is large (~2GB). You can skip it and use only Random Forest:
```powershell
# Install everything except torch
C:\Python313\python.exe -m pip install pyyaml numpy pandas scikit-learn matplotlib seaborn tqdm requests jsonlines radon plotly networkx
```

Then comment out LSTM-related code in `main.py` (lines with `LSTMPipeline`).

## What Gets Installed

| Package | Size | Purpose |
|---------|------|---------|
| pyyaml | Small | Configuration files |
| numpy | Medium | Numerical computing |
| pandas | Medium | Data manipulation |
| scikit-learn | Medium | ML algorithms |
| matplotlib | Medium | Visualization |
| seaborn | Small | Statistical plots |
| torch | **Large (2GB)** | Deep learning (LSTM) |
| transformers | Large | CodeBERT models |
| Others | Small | Utilities |

**Total size:** ~3-4 GB

## Quick Start After Installation

1. ✅ Install dependencies (you're here)
2. 📁 Prepare your data in `data/raw/`
3. ▶️ Run: `python main.py --step all`
4. 📊 Check results in `results/`

## Need Help?

- Check `QUICKSTART.md` for usage examples
- Check `README.md` for full documentation
- Check `PROJECT_SUMMARY.md` for overview

---

**Quick Command Reference:**

```powershell
# Install everything
.\install_quick.bat

# Run pipeline
C:\Python313\python.exe main.py --step all

# Run Jupyter notebooks
jupyter notebook notebooks/
```
