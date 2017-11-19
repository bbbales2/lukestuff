#%%

import skimage.io
import os
import pandas
import matplotlib.pyplot as plt
import numpy
import skimage.transform

os.chdir('/home/bbales2/lukestuff')

import chanvese

im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/GTD444_1229_2pct_10000x_Transverse_DendriteCore.TIF').astype('float'))[:1600, :1600]

im = skimage.transform.resize(im, (256, 256))

mean = numpy.mean(im.flatten())
std = numpy.std(im.flatten())

im = (im - mean) / std

phi = chanvese.segment(im, 400)
#%%
plt.imshow(im, cmap = plt.cm.gray)
#plt.imshow(phi < 0.0, alpha = 0.25)
plt.show()