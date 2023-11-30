# Installation
The requirement.txt file is friendly provided. You can prepare the environment by running `pip install -r requirements.txt` 
in your command line with python=3.8.

To avoid unpredictable mistakes, you can also install python=3.8, torch=1.13.1 and other packages separately.

# Inference using checkpoints

You can run the following codes to generate images using the checkpoint placed in `./ckpts`.

```
cd codes
python Main.py --state eval --ckpt ckpt_best.pt
```

The generated images are saved in `./codes/SampledImgs`. Note that it may take several seconds for the model to complete the denoising process.

# Train your own DDPM
Run the following codes to train your own DDPM.
```
cd codes
python Main.py --state train
```

The checkpoints will be saved in `./ckpts`.

# References

[Code Base](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)

[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672.pdf)

[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)
