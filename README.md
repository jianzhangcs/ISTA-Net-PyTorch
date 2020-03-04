# ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing [PyTorch version]
This repository is for ISTA-Net and ISTA-Net<sup>+</sup> introduced in the following paper

[Jian Zhang](http://jianzhang.tech/), [Bernard Ghanem
](http://www.bernardghanem.com/), "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing", CVPR 2018, [[pdf]](https://ivul.kaust.edu.sa/Documents/Publications/2018/ISTA-Net%20Interpretable%20Optimization-Inspired%20Deep%20Network%20for%20Image.pdf)

The code is built on **PyTorch** and tested on Ubuntu 16.04/18.04 and Windows 10 environment (Python3.x, PyTorch>=0.4) with 1080Ti GPU.

[old Tensorflow version](https://github.com/jianzhangcs/ISTA-Net)

## Contents
1. [Test](#test)
2. [Train](#train)
3. [Results](#results)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)


## Test
### Quick start
1. All models for our paper have been put in './model'.

2. Run the following scripts to test ISTA-Net models.

    **You can use scripts in file 'TEST_ISTA_Net_plus_scripts.sh' to produce results for our paper.**

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



## Train
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
