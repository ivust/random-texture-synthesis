# random-texture-synthesis
Code for the ICLR 2017 paper [“What does it take to generate natural textures?”](https://openreview.net/forum?id=BJhZeLsxx)

# How to run
The main function is `synthesise.py`, which allows to synthesise textures using random shallow models described in the paper.

Running

`python3 synthesise.py -t texture.jpg -s 200`

will synthesise a new texture of size 200x200 using the multi-scale model described in the paper and save it as `texture_synthesised.jpg`. You can change the sizes of convolutional filters for different scales using the “--scales” keyword. For example,

`python3 synthesise.py -t texture.jpg -s 200 --scales 11`

will synthesise a new texture using a single-scale model with 11x11 filters. Run `python3 synthesise.py -h` to see how to change number of optimisation steps or the number of feature maps per scale.
