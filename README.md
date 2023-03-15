# DPM-OT

**DPM-OT: A New Diffusion Probabilistic Model Based on Optimal Transport**

(Official Pytorch Implemention, the code is modified from [ncsnv2](https://github.com/ermongroup/ncsnv2))

## Introduction
This repo provide a fast diffusion probabilistic models (DPM) method which can generate high-quality samples within 5-10 function evaluations. 

Sampling from diffusion probabilistic models (DPMs) can be viewed as a piecewise distribution transformation, which generally requires hundreds or thousands of steps of the inverse diffusion trajectory to get a high-quality image. Recent progress in designing fast samplers for DPMs achieves a trade-off between sampling speed and sample quality by knowledge distillation or adjusting the variance schedule or the denoising equation. However, it can’t be optimal in both aspects and often suffer from mode mixture in short steps. To tackle this problem, we innovatively regard inverse diffusion as an optimal transport (OT) problem between latents at different stages and propose DPM-OT, a unified learning framework for fast DPMs with the direct expressway represented by OT map, which can generate high-quality samples within around 10 function evaluations. By calculating the semi-discrete optimal transmission between the data latents and the white noise, we obtain the expressway from the prior distribution to the data distribution, while significantly alleviating the problem of mode mixture. In addition, we give the error bound of the proposed method, which theoretically guarantees the stability of the algorithm.
## Results


## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/cognaclee/DPM-OT
    cd DPM-OT
    ```

2. Sampling

    First, generate the target latent variable through the following instructions.

    ```bash
    python main.py --test --config cifar10.yml -i cifar10 --doc cifar10 
    ```
    Then, calculate the OT map and sample the image with the following instructions.
    ```bash
    python main.py --sample --config cifar10.yml -i cifar10 --doc cifar10
    ```
    The above instructions are just examples of cifar10 datasets. For celeaba and ffhq, just replace cafar10 with the corresponding dataset name.

3. Evaluation

    3.1 improved precision and recall
    
    First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use ImageNet 256x256, so the refernce batch is `VIRTUAL_imagenet256_labeled.npz` and we can use the sample batch `admnet_guided_upsampled_imagenet256.npz`. Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](./metrics/requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). 
    Calculate the improved precision and recall metricby running the following script
     ```bash
     python ./metrics/evaluator.py VIRTUAL_imagenet256_labeled.npz admnet_guided_upsampled_imagenet256.npz
    ```

    Calculate the mode mixture metric MMR by running the following
    ### Train Model
    CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

    The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`.

    Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
    ```bash
    python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
    ```
    In addition, the model can also be trained on data set Tiny-Imagenet-200, and the call code is
    ```bash
    python3 train.py --name tiny_imagenet-200_500 --dataset tiny_imagenet_200 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
    ```
    However, this code does not automatically download Tiny-ImageNet-200 data set, so it is necessary to download the data set in advance and put it into 
    the 'data' folder before use, and then execute this code to train the model.

    ### Calculate the mode mixture metric MMR
    
    After training the model on the corresponding data set, we can run the following code (just take the Cifar10 data set as an example, other data sets are similar)
     to test the model on other images and output the number of mode mixture images.
    ```bash
    python3 test.py --name cifa10-test --trained_model_dir output/cifar10-100_500_checkpoint.pth --figure_dir figure/cifar10
    ```

## Citation

```
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)
```
