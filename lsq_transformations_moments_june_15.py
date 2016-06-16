#%%

import svgwrite
import os
import cairosvg
import tempfile
import skimage.io
import skimage.transform
import subprocess
import numpy
import time
import pickle
import skimage.feature
import sklearn.cross_validation
import matplotlib.patches
import sklearn.svm
import matplotlib.pyplot as plt
import skimage.filters
import sys
import skimage.measure
import mahotas
import itertools

os.chdir('/home/bbales2/microstructure_python')

import microstructure.features

os.chdir('/home/bbales2/lukestuff')
#%%
class Moments(object):
    def __init__(self):
        return

    def fit(self, ims):
        features = []

        for im in ims:
            features.append(self._internal_features(im))

        features = numpy.concatenate(features)

        self.bin_edges = []
        for i in range(features.shape[1]):
            _, bin_edges = numpy.histogram(features[:, i], 20)

            self.bin_edges.append(bin_edges)

        return self.bin_edges

    def compute(self, ims):
        Xs = []

        for im in ims:
            xs = []

            features = self._internal_features(im)

            for i in range(features.shape[1]):
                xs.extend(numpy.histogram(features[:, i], bins = self.bin_edges[i])[0])

            Xs.append(xs)

        return numpy.array(Xs)

    def _internal_features(self, im):
        sys.stdout.flush()
        thresh = skimage.filters.threshold_otsu(im)

        labeled, seeds = mahotas.label(im > thresh)
        labeled = mahotas.labeled.remove_bordering(labeled)

        features = []
        for seed in range(1, seeds):
            precipitates = (labeled == seed).astype('int')

            m00 = mahotas.moments(precipitates, 0, 0)
            m01 = mahotas.moments(precipitates, 0, 1)
            m10 = mahotas.moments(precipitates, 1, 0)
            m11 = mahotas.moments(precipitates, 1, 1)
            m02 = mahotas.moments(precipitates, 0, 2)
            m20 = mahotas.moments(precipitates, 2, 0)

            xm = m10 / m00
            ym = m01 / m00

            u00 = 1.0
            u11 = m11 / m00 - xm * ym
            u20 = m20 / m00 - xm ** 2
            u02 = m02 / m00 - ym ** 2

            w1 = u00**2 / (2.0 * numpy.pi * (u20 + u02))
            w2 = u00**4 / (16.0 * numpy.pi * numpy.pi * (u20 * u02 - u11**2))

            if numpy.isnan(w1) or numpy.isnan(w2) or numpy.isinf(w1) or numpy.isinf(w2):
                continue

            features.append([numpy.sqrt(m00), w1, w2, u20, u02])#

        features = numpy.array(features)

        return features
#%%

#%%
tfs = []
#tfs1 = []
#ims = []
for y in range(0, 8):
    ims = []
    dups = 8
    for r in range(dups):##rafting2a1h5_rotatedrafting2a1h5
        im = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(y, r * 8), as_grey = True).astype('double')
        #print im.shape
        im -= numpy.mean(im.flatten())
        im /= numpy.std(im.flatten())

        ims.append(im)

    tfs.append((ims, y))
#%%
moments = Moments()

out = moments.fit(tfs[0][0] + tfs[-1][0])

Xs = moments.compute(itertools.chain(*[t[0] for t in tfs]))
Ys = list(itertools.chain(*[[t[1]] * len(t[0]) for t in tfs]))
#%%
#%%
#for r in range(dups):#nrafting2anrafting2arafting2a1h5
#    tfs1.append('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(8, r * 8))

def extract_features(filenames, get_features):
    Xs = []

    for filename in filenames:
        im = skimage.io.imread(filename, as_grey = True).astype('double')
        #print im.shape
        im -= numpy.mean(im.flatten())
        im /= numpy.std(im.flatten())

        #plt.imshow(im)
        #plt.show()

        Xs.append(get_features(im))

    return Xs

f0s_ = []
for filenames, y in tfs:
    f0s = extract_features(filenames, moment_features)
    f0s_.append(f0s)
#%%
Xs = []
Ys = []
for f0s, (filenames, y) in zip(f0s_, tfs):
    for f0 in f0s:
        hist0, _ = numpy.histogram(f0[:, 0], bins = bin_edges)
        hist1, _ = numpy.histogram(f0[:, 1], bins = bin_edges)

        Xs.append(numpy.concatenate((hist0, hist1))
        Ys.append(y)

    print y

#f1s = extract_features(tfs1, moment_features)
#%%
for X in Xs:
    plt.plot(X)
plt.show()
#%%
ridge = sklearn.linear_model.Ridge()

#print sklearn.cross_validation.cross_val_score(ridge, Xs, Ys)
ridge.fit(Xs, Ys)

plt.plot(Ys)
plt.plot(ridge.predict(Xs))
