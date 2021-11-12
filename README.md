# UNIFORMGENERALIZATIONBOUNDS  FOROVERPA-RAMETERIZEDNEURALNETWORKS
This repository is the official implementation of *Uniform Generalization Bounds for Overparametrized Neural Network*.
Adopting therecently developed Neural Tangent (NT) kernel theory, we prove uniform gener-alization bounds for overparameterized neural networks in kernel regimes, whenthe true data generating model belongs to the reproducing kernel Hilbert space(RKHS) corresponding to the NT kernel. Importantly, our bounds capture the exact error rates depending on the differentiability of the activation functions

This anonymous repository is constructed to calculate the error rates vs dataset size  for different activatin functions. 

## Requirements
This repository has been developed with python3.7 and pip 21.3.1. Using other python and pip versions have been proven problematic when installing some of the required libraries.
To install requirements:
```
pip install jax jaxlib --upgrade
pip install neural-tangents
pip install matplotlib jupyter
```
## Notebook
The attached jupyter notebook will allow you to:
- Calculate the error rate for a toy example
- Reproduce the plots in fig 2 of the paper. 


