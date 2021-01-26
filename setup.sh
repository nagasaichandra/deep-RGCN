#!/usr/bin/env bash
pip install tensorboard
pip install torch==1.7.1
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html
pip install torch-geometric
#pip install requests
#pip install tqdm