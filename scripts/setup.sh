#!/bin/bash
# environment configuration script (Bash)
# usage: bash setup.sh in the FlowInOne directory

echo "=== Step 1/4: install PyTorch (CUDA 12.1) ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 2/4: install main dependencies ==="
pip install -r requirements.txt

echo "=== Step 3/4: install pycocotools ==="
pip install --no-build-isolation "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

echo "=== Step 4/4: install webdataset ==="
pip install git+https://github.com/tmbdev/webdataset.git

echo "=== installation completed! ==="