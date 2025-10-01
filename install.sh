#!/bin/bash

python3 -m venv iem
source iem/bin/activate
pip install --upgrade pip

pip install dctorch
pip install einops

# Change the cuda versions according to your system
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install natten==0.21.0+torch270cu128 -f https://whl.natten.org

