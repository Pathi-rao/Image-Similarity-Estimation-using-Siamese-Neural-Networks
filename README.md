# MVTEC ANOMALY DETECTION USING SIAMESE NEURAL NETWORKS

<!-- TABLE OF CONTENTS -->
## Table of Contents
  - [About the Project](#about-the-project)
  - [Installation](#installation)
  - [References](#references)

## About the Project
This project is aimed to implement a image similarity estimation model using deep learning that will be able to generalize the difference between two given images.

We have used the [dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) from MVTec AD which contains good and anomaly samples from 15 classes and trained the model using Siamese Neural Networks. We were able to do that by pairing images from same and different models such that the model can learn the similarity metric. 



## Installation

1. Clone the repo using the following command:
```bash
git clone https://github.com/Pathi-rao/Project_for_Numerical_methods_for_algorithmic_systems_and_NeuralNetworks 
```
2. Create a virtual environment with Python(For this step I will assume that you are able to create a virtual environment with `virtualenv`, but in any case you can check an example [here](https://realpython.com/python-virtual-environments-a-primer/).)

 - Activate the virtual environment and `cd` into the project directory

3. Install requirements using `pip`:
```bash
pip install -r requirements.txt
```
4. Run `train.py` to start training the model and use `test.py` to do inference on any two images
## References

* [PyTorch](https://github.com/pytorch/pytorch) for training deep-learning models.

