# DPM-OT

**DPM-OT: A New Diffusion Probabilistic Model Based on Optimal Transport**

(Official Pytorch Implemention, the code is modified from [ncsnv2](https://github.com/ermongroup/ncsnv2))

## Introduction
<!Sampling from diffusion probabilistic models (DPMs) can be viewed as a piecewise distribution transformation, which generally requires hundreds or thousands of steps of the inverse diffusion trajectory to get a high-quality image. Recent progress in designing fast samplers for DPMs achieves a trade-off between sampling speed and sample quality by knowledge distillation or adjusting the variance schedule or the denoising equation. However, it can’t be optimal in both aspects and often suffer from mode mixture in short steps. To tackle this problem, we innovatively regard inverse diffusion as an optimal transport (OT) problem between latents at different stages and propose DPM-OT, a unified learning framework for fast DPMs with the direct expressway represented by OT map, which can generate high-quality samples within around 10 function evaluations. By calculating the semi-discrete optimal transmission between the data latents and the white noise, we obtain the expressway from the prior distribution to the data distribution, while significantly alleviating the problem of mode mixture. In addition, we give the error bound of the proposed method, which theoretically guarantees the stability of the algorithm.>
A Fast diffusion probabilistic models (DPM) method which can generate high-quality samples within 5-10 function evaluations. 

## Results


## Installation

* Create conda virtual environment


## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/cognaclee/DPM-OT
    cd DPM-OT
    ```

2. Training

    ```bash
    python 
    ```

3. Testing


    ```bash
    python 
    ```

4. Evaluation

    ```bash


## Citation

```
ggg
```
