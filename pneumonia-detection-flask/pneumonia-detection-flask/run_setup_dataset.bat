@echo off
REM Download Kaggle Chest X-Ray Pneumonia dataset into dataset/train|val|test
cd /d "%~dp0"
echo Installing kaggle if needed...
pip install kaggle -q
python setup_dataset.py
if errorlevel 1 (
  echo.
  echo If download failed: add your Kaggle API key to %%USERPROFILE%%\.kaggle\kaggle.json
  echo Get key from: https://www.kaggle.com/settings ^> Create New Token
  echo Accept rules: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  pause
)
