#!/bin/bash

# 确保存在所有必要的系统依赖
sudo apt-get update
sudo apt-get install -y libgomp1 libstdc++6

# 强制重新安装 SHAP
pip install --no-cache-dir --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  https://github.com/slundberg/shap/releases/download/v0.45.1/shap-0.45.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 验证安装
python -c "import shap; print(f'SHAP version: {shap.__version__}')"
