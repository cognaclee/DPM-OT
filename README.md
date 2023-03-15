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

    Calculate the improved precision and recall metricby running the following script
     ```bash
     python evaluator.py VIRTUAL_imagenet256_labeled.npz admnet_guided_upsampled_imagenet256.npz
    ```
    First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use ImageNet 256x256, so the refernce batch is `VIRTUAL_imagenet256_labeled.npz` and we can use the sample batch `admnet_guided_upsampled_imagenet256.npz`. Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

    Calculate the mode mixture metric MMR by running the following
    ```bash
    python3 test.py --name cifa10-test --trained_model_dir output/cifar10-100_500_checkpoint.pth --figure_dir figure/cifar10
    ```


## Citation

```
...
```
