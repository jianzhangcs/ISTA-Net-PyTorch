# ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing [PyTorch version]

## including codes of CS for natural image (CS-NI) and CS for magnetic resonance imaging (CS-MRI) 

This repository is for ISTA-Net and ISTA-Net<sup>+</sup> introduced in the following paper

[Jian Zhang](http://jianzhang.tech/), [Bernard Ghanem
](http://www.bernardghanem.com/), "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing", CVPR 2018, [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ISTA-Net_Interpretable_Optimization-Inspired_CVPR_2018_paper.pdf)

The code is built on **PyTorch** and tested on Ubuntu 16.04/18.04 and Windows 10 environment (Python3.x, PyTorch>=0.4) with 1080Ti GPU.

[[Old Tensorflow Version]](https://github.com/jianzhangcs/ISTA-Net)

## Introduction
With the aim of developing a fast yet accurate algorithm for compressive sensing (CS) reconstruction of natural images, we combine in this paper the merits of two existing categories of CS methods: the structure insights of traditional optimization-based methods and the speed of recent network-based ones. Specifically, we propose a novel structured deep network, dubbed ISTA-Net, which is inspired by the Iterative Shrinkage-Thresholding Algorithm (ISTA) for optimizing a general L1 norm CS reconstruction model. To cast ISTA into deep network form, we develop an effective strategy to solve the proximal mapping associated with the sparsity-inducing regularizer using nonlinear transforms. All the parameters in ISTA-Net (\eg nonlinear transforms, shrinkage thresholds, step sizes, etc.) are learned end-to-end, rather than being hand-crafted. Moreover, considering that the residuals of natural images are more compressible, an enhanced version of ISTA-Net in the residual domain, dubbed ISTA-Net+, is derived to further improve CS reconstruction. Extensive CS experiments demonstrate that the proposed ISTA-Nets outperform existing state-of-the-art optimization-based and network-based CS methods by large margins, while maintaining fast computational speed.

![ISTA-Net](/Figs/ista_phase.png)
Figure 1. Illustration of the proposed ISTA-Net framework.


## Contents
1. [Test-CS-NI](#test-cs-ni)
2. [Train-CS-NI](#train-cs-ni)
3. [Test-CS-MRI](#test-cs-mri)
4. [Train-CS-MRI](#train-cs-mri)
5. [Results](#results)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)


## Test-CS-NI
### Quick start
1. All models for our paper have been put in './model'.

2. Run the following scripts to test ISTA-Net models.

    **You can use scripts in file 'TEST_ISTA_Net_scripts.sh' to produce results for our paper.**

    ```bash
    # test scripts
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 1 --layer_num 9
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 4 --layer_num 9
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 10 --layer_num 9
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 25 --layer_num 9
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 30 --layer_num 9
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 40 --layer_num 9
    python TEST_CS_ISTA_Net.py --epoch_num 200 --cs_ratio 50 --layer_num 9
    ```

3. Run the following scripts to test ISTA-Net<sup>+</sup> models.

    **You can use scripts in file 'TEST_ISTA_Net_plus_scripts.sh' to produce results for our paper.**

    ```bash
    # test scripts
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 1 --layer_num 9
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 4 --layer_num 9
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 10 --layer_num 9
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 25 --layer_num 9
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 30 --layer_num 9
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 40 --layer_num 9
    python TEST_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 50 --layer_num 9
    ```

### The whole test pipeline
1. Prepare test data.

    The original test set11 is in './data'

2. Run the test scripts. 

    See **Quick start**
3. Check the results in './result'.



## Train-CS-NI
### Prepare training data  

1. Trainding data (**Training_Data.mat** including 88912 image blocks) is in './data'. If not, please download it from [GoogleDrive](https://drive.google.com/file/d/14CKidNsC795vPfxFDXa1FH9QuNJKE3cp/view?usp=sharing) or [BaiduPan [code: xy52]](https://pan.baidu.com/s/1X3pERjCD37YdqQuzKNXejA).

2. Place **Training_Data.mat** in './data' directory

### Begin to train


1. run the following scripts to train ISTA-Net models.

    **You can use scripts in file 'Train_ISTA_Net_scripts.sh' to train models for our paper.**

    ```bash
    # CS ratio 1, 4, 10, 25, 30, 40, 50
    # train scripts
    python Train_CS_ISTA_Net.py --cs_ratio 10 --layer_num 9
    python Train_CS_ISTA_Net.py --cs_ratio 25 --layer_num 9
    python Train_CS_ISTA_Net.py --cs_ratio 50 --layer_num 9
    python Train_CS_ISTA_Net.py --cs_ratio 1 --layer_num 9
    python Train_CS_ISTA_Net.py --cs_ratio 4 --layer_num 9
    python Train_CS_ISTA_Net.py --cs_ratio 30 --layer_num 9
    python Train_CS_ISTA_Net.py --cs_ratio 40 --layer_num 9
    ```
    
    **We found that the re-trained ISTA-Net models may get a bit higher performance than the results reported in our paper.** 
    
2. run the following scripts to train ISTA-Net<sup>+</sup> models.
    
    **You can use scripts in file 'Train_ISTA_Net_plus_scripts.sh' to train models for our paper.** 

    ```bash
     # CS ratio 1, 4, 10, 25, 30, 40, 50
    # train scripts
    python Train_CS_ISTA_Net_plus.py --cs_ratio 10 --layer_num 9
    python Train_CS_ISTA_Net_plus.py --cs_ratio 25 --layer_num 9
    python Train_CS_ISTA_Net_plus.py --cs_ratio 50 --layer_num 9
    python Train_CS_ISTA_Net_plus.py --cs_ratio 1 --layer_num 9
    python Train_CS_ISTA_Net_plus.py --cs_ratio 4 --layer_num 9
    python Train_CS_ISTA_Net_plus.py --cs_ratio 30 --layer_num 9
    python Train_CS_ISTA_Net_plus.py --cs_ratio 40 --layer_num 9
    ```



## Test-CS-MRI
### Quick start
1. All models for our paper have been put in './model'.

2. Run the following scripts to test ISTA-Net<sup>+</sup> models.

    ```bash
    # test scripts
    python TEST_MRI_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 20 --layer_num 9
    python TEST_MRI_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 30 --layer_num 9
    python TEST_MRI_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 40 --layer_num 9
    python TEST_MRI_CS_ISTA_Net_plus.py --epoch_num 200 --cs_ratio 50 --layer_num 9
    ```

### The whole test pipeline
1. Prepare test data.

    The original test BrainImages_test is in './data'

2. Run the test scripts. 

    See **Quick start**
3. Check the results in './result'.



## Train-CS-MRI

### Prepare training data  

1. Trainding data (**Training_BrainImages_256x256_100.mat** including 88912 image blocks) is in './data'. If not, please download it from [GoogleDrive](https://drive.google.com/file/d/1Gh2pKtVosXmGyv26rERihU27f9UKjcmz/view?usp=sharing).

2. Place **Training_BrainImages_256x256_100.mat** in './data' directory

### Begin to train


1. run the following scripts to train ISTA-Net<sup>+</sup> models.
    
    **You can use scripts in file 'Train_ISTA_Net_plus_scripts.sh' to train models for our paper.** 

    ```bash
    # train scripts
    python Train_MRI_CS_ISTA_Net_plus.py --cs_ratio 20 --layer_num 9
    python Train_MRI_CS_ISTA_Net_plus.py --cs_ratio 30 --layer_num 9
    python Train_MRI_CS_ISTA_Net_plus.py --cs_ratio 40 --layer_num 9
    python Train_MRI_CS_ISTA_Net_plus.py --cs_ratio 50 --layer_num 9
    ```

## Results
### Quantitative Results
### Visual Results

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{zhang2018ista,
  title={ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing},
  author={Zhang, Jian and Ghanem, Bernard},
  booktitle={CVPR},
  pages={1828--1837},
  year={2018}
}
```
## Acknowledgements