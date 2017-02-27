# random-texture-synthesis
Code for the ICLR 2017 paper [“What does it take to generate natural textures?”](https://openreview.net/forum?id=BJhZeLsxx)

# How to run
## Synthesising textures
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

* `-t` is a path to a reference textures,
* `-s` is a size (in pixels) of a synthesised textures,
* `-f` is a filename of a synthesised texture (if not provided, by default `_synthesised.jpg` is added to the filename of the reference texture),
* `-n` is a number of L-BFGS optimisation steps (default 4000),
* `-c` is a number of feature maps per scale (i.e. for each filter size),
* `--scales` are the sizes of convolutional filters (if not provided the multi-scale model is used by default with filters sizes 3, 5, 7, 11, 15, 23, 37, 55,
* `-l` makes the model linear (no non-linearity after the conv layer is used).

## Evaluating VGG-loss
`vgg_loss.py` allows to evaluate the VGG-loss of a synthesised texture. Function `main` in this file takes two arguments (path to the reference texture and path to the synthesised texture) and returns the value of the VGG-loss:

```python
import vgg_loss

loss = vgg_loss.main('reference_texture.jpg', 'synthesised_texture.jpg')
```

# Figures
Folders for respective figures contain code to generate these figures using the provided synthesised textures as well as the shell scripts (e.g. `synthesise_figure2.sh`) to generate new textures.
