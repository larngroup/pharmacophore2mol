"""
This module contains my implementation of a 3D UNet model based on the diffusers library.
@maryam here is a retry for the 3D UNet implementation
The need for this module arises from the diffusers library's lack of a 3D UNet model.
In case you're wondering why I didn't use the existing UNet3DConditionModel,
even though it seems to be a 3D model, it is actually a 2D model with a temporal dimension.
It processes 2D spatial data with a temporal dimension, aka video, not true 3D spatial data.
Therefore, I had to implement my own 3D UNet model from scratch.
I will base this implementation as closely as possible on the existing 2D UNet architecture,
basically copying and adapting where needed. I will keep the defaults of the 2d unet, as it
worked well for 2d data. Maybe an increase in model size may e needed, but i'll check that later.
This sits on the random folder until I can figure out how to integrate it with the rest of the code.
"""