# GreensONet
This repository provides the code and data for following research papers:  
Jianghang Gu, Ling Wen, Yuntian Chen, and Shiyi Chen, An explainable operator approximation framework under the guideline of Green's function.


# Framework
The framework of GreensONet:

(a) Import user-defined free tetrahedral mesh and user-defined physical conditions;

(b) Calculate the locations of Gauss integration points and integration weights; 

(c) Constructions of the Trunk Net and Branch Net of the GreensONet based on binary structured neural networks; 

(d) Domain partition and parallel computation strategy; 

(e) Volterra integration based on acquired Green's function; 

(f) The calculated solutions. 

![framework](workflow.jpg)


# Installation
```
conda create -n GreensONet python=3.10 # Create a Python 3 virtual environment with conda.
conda activate GreensONet # Activate the virtual environment
pip install -r requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Download Data and Pretrained Checkpoints

PaddleScience (Recommend)
```
mkdir data
cd data
wget https://dataset.bj.bcebos.com/PaddleScience/PaddleCFD/GON_dataset.tar
tar -xvf GON_dataset.tar
cd ..
wget https://dataset.bj.bcebos.com/PaddleScience/PaddleCFD/ckpt.tar
tar -xvf ckpt.tar
```

Google Drive:

[data](https://drive.google.com/file/d/1lv0WuWCrZsB2MZcaMDFRn06HsxGLgNaq/view?usp=sharing)

[pretrain checkpoint](https://drive.google.com/file/d/122U51gwpriZyQKg6WPCaSXCYX8DzOAIV/view?usp=sharing)

# Test
```
python Inference.py
```

# Train
```
cd GreenONet
chmod 777 -R run.sh
./run.sh
```


# Reference

(1) https://github.com/hangjianggu/Discover_Green_function/tree/main

(2) https://github.com/sloooWTYK/GF-Net/tree/main

(3) https://greenlearning.readthedocs.io/en/latest/guide/installation.html


# [Case 1: Heterogeneous reaction-diffusion equations] [name:Flat Plane]
## Paddle
MSE loss over [25] test cases 1.44e-03
MSE with BC loss over [25] test cases 7.33e-04
25 test, 80 train

## Torch
8.28e-4 over [5] test cases

## Paper
Table 4: The hyper-parameters and performance of different baseline models on case of flat plane.
2000, 0.001, [6, 12, 12, 12, 1], 2.31 × 10−4, 5.22 × 10−4
30 test, 70 train

# [Case 2: Steady heat conduction equations] [name : Finned Tube]
## Paddle:
MSE loss over [20] test cases 1.47e-03
MSE with BC loss over [20] test cases 2.29e-05
20 test, 80 train

## Torch
2.623e-05 over [20] test cases

Table 3: The hyper-parameters and performance of different baseline models on case of finned tube.
2000, 0.001, [6, 12, 12, 12, 1], 2.18 × 10−5, 2.64 × 10−5
30 test, 70 train

# [Case 3: Stokes equations] [name : 3D lid-driven cavity]
## Paddle:
MSE loss over [5] test cases 1.58e-03
MSE with BC loss over [5] test cases 8.65e-04
5 test, 15 train

## Torch(Cant run, convert to Paddle checkpoint and infer)
8.65e-04 over [20] test cases

## Paper
Table 6: The training and model hyper-parameters of different baseline models.
7000, 0.001, [6, 24, 24, 24, 1], 4.43 × 10−4, 4.63 × 10−4
? test, ? train

# [Case 4: Heterogeneous reaction-diffusion equations] [name : 3D finned tube]
## Paddle:
MSE loss over [5] test cases 8.16e-04
MSE with BC loss over [5] test cases 4.66e-04

## Torch
4.62 × 10−4 over [5] test cases

## Paper
Table 5: The hyper-parameters and performance of different baseline models on case of pipe.
GON, 7000, 0.001, [6, 24, 24, 24, 1], 4.43 × 10−4, 4.63 × 10−4
? test, ? train
