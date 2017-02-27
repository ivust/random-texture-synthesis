# random-texture-synthesis
Code for the ICLR 2017 paper [“What does it take to generate natural textures?”](https://openreview.net/forum?id=BJhZeLsxx)

# How to run
The main function is `synthesise.py`, which allows to synthesise textures using random shallow models described in the paper.

Running

`python3 synthesise.py -t texture.jpg -s 200`

will synthesise a new texture of size 200x200 using the multi-scale model described in the paper and save it as `texture_synthesised.jpg`. You can change the sizes of convolutional filters for different scales using the “--scales” keyword. For example,

`python3 synthesise.py -t texture.jpg -s 200 --scales 11`

will synthesise a new texture using a single-scale model with 11x11 filters. 

Additional arguments to `synthesise.py` are as follows:

```
python3 synthesise.py -h

usage: synthesise.py [-h] -t TEXTURE -s SIZE [-f TARGET_FILE] [-n N_ITER]
                     [-c N_FEATURES] [--scales [SCALES [SCALES ...]]] [-l]

optional arguments:
  -h, --help            show this help message and exit
  -t TEXTURE, --texture TEXTURE
                        Path to the reference texture
  -s SIZE, --size SIZE  Size of a synthesised texture in pixels
  -f TARGET_FILE, --target-file TARGET_FILE
                        File name of a syntesised texture
  -n N_ITER, --n-iter N_ITER
                        Number of L-BFGS optinisation iterations
  -c N_FEATURES, --n-features N_FEATURES
                        Number of feature maps per each scale
  --scales [SCALES [SCALES ...]]
                        Sizes of convolutional filters
  -l, --linear          Use linear model (conv layer without non-linearity)
```
