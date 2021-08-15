# MultiResViT

This is the repository for the model MultiResViT, our proposed model architecture for image classification in the medical domain. 

## Dataset

We used three of the ten available MedMNIST datasets for our experiments. We stored the datasets (both super-resolved and regular sized images) in h5 files, the links to which we make available [here](https://drive.google.com/drive/folders/1J1OeMI2e4ym-ZWfdifXS2oe9wXZwL3rW?usp=sharing). The keys for the datasets are `['label_test', 'label_train', 'test', 'test_small', 'train', 'train_small']`. `train` and `test` contain the super-resolved images, while `train_small` and `test_small` contain the original images.

## For using the code

This repository contains a notebook called MRViT that is built as a starter for running the MRViT model on any of the three given datasets. Just edit the `dataset` variable and the number of classses as per your convenience. The NesT and LeViT models have both been included in the repo for experimentation purposes - these borrow heavily from the [lucidrains](https://github.com/lucidrains/vit-pytorch) implementation of the same. The API is as follows:

```python
"""
  Example code for the NesT model API.
"""
nest = NesT(
    image_size = 224,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (8, 4, 1),  # the number of transformer blocks at each hierarchy, starting from the bottom
    num_classes = 1000
)

"""
  Example code for the LeViT model API.
"""
levit = LeViT(
    image_size = 224,
    num_classes = 1000,
    stages = 3,             # number of stages
    dim = (256, 384, 512),  # dimensions at each stage
    depth = 4,              # transformer of depth 4 at each stage
    heads = (4, 6, 8),      # heads at each stage
    mlp_mult = 2,
    dropout = 0.1
)
```
