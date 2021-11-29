## MTTU-Net
Code for "A Fully Automated Multimodal MRI-based Multi-task Learning for Glioma Segmentation and IDH Genotyping"
## Prerequisites
Python 3.7+

Pytorch 1.7.0+

This code has been tested with Pytorch 1.7.0 and two NVIDIA V100 GPU.

## Data descriptions
The data used in this study includes the MRI data and IDH genomic information. MRI data are derived from [BraTS 2020](https://ipp.cbica.upenn.edu/) and [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/), and the corresponding genomic information is from [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/). We have provided the mapping about this data.

## Training the model
python -m torch.distributed.launch --nproc_per_node=2 --master_port 22 train_IDH.py

## Inference
python test.py
