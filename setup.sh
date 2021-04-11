#!/usr/bin/env bash
pip install logging
pip install tensorboard
pip install torch==1.4.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
pip install torch-geometric
pip install requests
pip install tqdm