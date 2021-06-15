# mac-network-pytorch
Memory, Attention and Composition (MAC) Network for CLEVR from Compositional Attention Networks for Machine Reasoning (https://arxiv.org/abs/1803.03067) implemented in PyTorch


To train:

1. Download and extract CLEVR v1.0 dataset and preprocess questions and images:
```
$ wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
$ python preprocess.py [CLEVR directory]
$ python image_feature.py [CLEVR directory]
```
!CAUTION! the size of file created by image_feature.py is very large! (~70 GiB) You may use hdf5 compression, but it will slow down feature extraction.

2. Run train.py (with arguments from config.yml or arguments.py)
```
python train.py --config main --dataset_root [CLEVR directory]
```

3. Run test.py (with arguments from config.yml or arguments.py)
```
python test.py --config main --dataset_root [CLEVR directory] --checkpoint_path [CHECKPOINT path]
```