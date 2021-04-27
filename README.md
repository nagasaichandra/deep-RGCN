# Deep RGCN

Source code for Deep RGCN model with dense and residual skip connections. The specifications of the model Deep RGCN can be found in the thesis -- "Node Classification on Relational Graphs using Deep-RGCNs" available for download at: https://digitalcommons.calpoly.edu/theses/2265/. In this work, the recent developments with respect to knowledge graphs and graph learning is reviewed and a generalized framework to allow training of deep neural networks to learn on node representations is proposed. This is possible by the use of dense and residual skip connection techniques adopted from popular CNN frameworks such as ResNet and DenseNet architectures. These special connections allow consistent training while preventing oversmoothing issues.

## Requirements
torch > 1.4.0
torch-geometric > 1.6.3, torch-scatter, torch-sparse, torchvision
logging
tensorboard

## Setup
just run the following command in the directory to install the required libraries:

`./setup.sh`

## Reproduction
To reproduce the model (Train and test on entities dataset of torch_geometric relational graphs):

`python main.py --dataset AIFB --block_type res --aggr mean`

(Note: The results may not always be exactly the same.)


