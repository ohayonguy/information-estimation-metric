
This repository contains the official PyTorch implementation of the Information-Estimation Metric (IEM) introduced in the paper:
<div align="center">

# [Learning a distance measure from the information-estimation geometry of data](https://arxiv.org/abs/2510.02514)<br />

[Guy Ohayon](https://ohayonguy.github.io/),&nbsp;&nbsp;[Pierre-Etienne H. Fiquet](https://www.cns.nyu.edu/~fiquet/),&nbsp;&nbsp;[Florentin Guth](https://florentinguth.github.io/),&nbsp;&nbsp;[Jona Ballé](https://balle.io/),&nbsp;&nbsp;[Eero P. Simoncelli](https://www.cns.nyu.edu/~eero/)<br />

</div>

### ⚙️ Installation
The IEM is computed using a diffusion model. Specifically, we use the [Hourglass Diffusion Transformer (HDiT)](https://github.com/crowsonkb/k-diffusion) architecture, which requires installing [NATTEN](https://github.com/SHI-Labs/NATTEN/tree/main).
You can install all the required packages using the provided `install.sh` file (adjust the cuda version according to your system).

Some Windows users may face an issue when installing the [NATTEN](https://github.com/SHI-Labs/NATTEN/tree/main) package. A solution that worked for some Windows users is suggested [here](https://github.com/ohayonguy/PMRF/issues/8#issue-2581034421).

### ⬇️ Download checkpoints

Download the diffusion model checkpoint from [Google Drive](https://drive.google.com/file/d/1sCHPTdYjbwTLsd5tPu5Zok28wloDkjXw/view?usp=sharing). Move the checkpoint to the `checkpoints/` folder.

### ⚡ Inference
See code below or `examples.ipynb`.
```
from information_estimation_metric import InformationEstimationMetric

# Load the IEM. You may use any other uncoditional diffusion model, but this will require some code adjustments.
iem = InformationEstimationMetric('./checkpoints/imagenet_256x256_loguniform_00400000.pth', 'bf16', True).cuda()

# Choose the number of numerical integration steps.
num_gamma = 64

# Choose the noise scale range.
sigma_min = 1
sigma_max = 1e3

# Choose the type of distance (standard IEM or generalized IEM with some "activation" function f).
iem_type = 'standard'

# Choose noise seed.
seed = 42

# Compute the IEM between a pair of images x1 and x2.
# To comply with the diffusion model we trained, the images must be of size 256x256 and have pixel values in the range [-1,1].
iem(x1, x2, num_gamma=num_gamma, sigma_min=sigma_min, sigma_max=sigma_max, iem_type=iem_type, seed=seed)
```

### 📈 Paper plots
The `paper_plots/` folder contains code to generate some of the figures in our paper.

### Optimizing the IEM under fixed PSNR
The `examples.ipynb` notebook contains an example where we maximize/minimize the IEM (and other metrics) between an image
$x$ and a distorted version $x+\epsilon$ while keeping the PSNR between them fixed (using projected gradient descent).
### 📝 Citation
```
@article{ohayon2025iem,
      title={Learning a distance measure from the information-estimation geometry of data}, 
      author={Guy Ohayon and Pierre-Etienne H. Fiquet and Florentin Guth and Jona Ballé and Eero P. Simoncelli},
      year={2025},
      journal={arXiv preprint arXiv:2510.02514},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2510.02514}, 
}
```
### 📋 License
This project is released under the [MIT license](https://github.com/ohayonguy/information-estimation-metric/blob/main/LICENSE).