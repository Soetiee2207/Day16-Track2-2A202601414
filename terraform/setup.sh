#!/bin/bash
mkdir -p ~/.kaggle
echo '{"username": "soetiee", "key": "KGAT_99b732c4ce6782dc67582b3b4c3c8487"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p ~/ml-benchmark
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/

python3 ~/benchmark.py
