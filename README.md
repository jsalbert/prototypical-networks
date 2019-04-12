# Prototypical-networks

This repository contains code implementing the Prototypical Networks for Few-Shot Learning [paper](https://arxiv.org/pdf/1703.05175.pdf). 

It is done in a style similar to [PyTorch Imagenet training script](https://github.com/pytorch/examples/tree/master/imagenet) allowing for data-parallel training, different architectures and datasets. Also I provide an `alpha` parameter that allows averaging the new prototypes with the last iteration ones. `class_prototypes = args.alpha * old_class_prototypes + (1 - args.alpha) * new_prototypes`

There is code for Mini-ImageNet dataset and code prepared for general usage. When testing on general usage, I use all the available samples as support. This part can be replaced for the Mini-ImageNet testing part if needed. 

Works with Pytorch 1.0.0 and python 3+.

## Code Usage

1. Install virtualenv, create environment and activate it

```
pip3 install virtualenv 
virtualenv .protonets 
source .protonets/bin/activate
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Modify the bash files with your selected options

## Run training and test on Mini-ImageNet

1. Download [Mini-ImageNet images]( https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and place them on folder `mini_imagenet/'

2. Default will run 1-Shot, 30-way training

`bash script_train_mini_imagenet.sh`

3. Default will test averaging over 2000 iterations

`bash script_test_mini_imagenet.sh`

## Acknowledgements

Based on the [original code](https://github.com/jakesnell/prototypical-networks) and [this](https://github.com/cyvius96/prototypical-network-pytorch) implementation.

## References 

```@article{DBLP:journals/corr/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  journal   = {CoRR},
  volume    = {abs/1703.05175},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.05175},
  archivePrefix = {arXiv},
  eprint    = {1703.05175},
  timestamp = {Wed, 07 Jun 2017 14:41:38 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/SnellSZ17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}```
