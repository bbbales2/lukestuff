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

os.chdir('/home/bbales2/microstructure_python')

import microstructure.features

os.chdir('/home/bbales2/lukestuff')

sys.path.append('/home/bbales2/caffe-tensorflow')

import googlenet

import tensorflow as tf
#%%
data = {}
for size in [32, 64, 128, 256]:
    dsets = {}
    for dset in ['nrafting2a', 'rafting2a1h5']:
        trains = {}
        tests = {}

        dups = 8
        for y in range(10):
            print size, dset, y

            samples = []
            for r in range(dups):##rafting2a1h5_rotated#
                im = skimage.io.imread('/home/bbales2/rafting/{0}/images_{1}/signal{2}.png'.format(dset, y, r * 4), as_grey = True)

                for i in range(0, im.shape[0], size):
                    for j in range(0, im.shape[1], size):
                        sample = im[i : i + size, j : j + size].astype('float')

                        sample -= numpy.mean(sample.flatten())
                        sample /= numpy.std(sample.flatten())

                        samples.append(sample)

            trains[float(y)] = samples

        for y in range(10):
            samples = []
            for r in range(dups):##rafting2a1h5_rotated#
                im = skimage.io.imread('/home/bbales2/rafting/{0}/images_{1}/signal{2}.png'.format(dset, y, 128 + r * 4), as_grey = True)

                for i in range(0, im.shape[0], size):
                    for j in range(0, im.shape[1], size):
                        sample = im[i : i + size, j : j + size].astype('float')

                        sample -= numpy.mean(sample.flatten())
                        sample /= numpy.std(sample.flatten())

                        samples.append(sample)

            tests[float(y)] = samples

        dsets[dset] = {
            'train' : trains,
            'test' : tests
        }

    data[size] = dsets
#%%
im = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(1, 0), as_grey = True)
im2 = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(1, 5), as_grey = True)
#%%
features = {}
for ftype, Features in (('Moments', Moments), ('NN', NN), ('LBP', LBP), ('HOG', HOG)):
    fdata = {}
    for size in [32, 64, 128, 256]:
        dsets = {}
        for dset in ['nrafting2a', 'rafting2a1h5']:
            trains = {}
            tests = {}

            for y in range(10):
                print ftype, size, dset, y
                b = size

                ft = Features()

                #features = Moments()
                ims = data[size][dset]['train'][y]

                ft.fit(ims)

                trains[y] = ft.get_features(ims)

                ims = data[size][dset]['test'][y]

                tests[y] = ft.get_features(ims)

            dsets[dset] = {
                'train' : trains,
                'test' : tests
            }
        fdata[size] = dsets
    features[ftype] = fdata

#%%
import pickle

f = open('lsqdata', 'w')
pickle.dump((data, features), f)
f.close()

#%%

b = 32
features = NN()

#features = Moments()
features.fit([im])

out = features.get_features([im, im])
print out[0].shape
#%%

b = 256

features = NN()

features.fit(data[b]['test'][0.0][0:32])

fs = features.get_features(data[b]['test'][0.0][0:32])

print fs[0].shape

#%%
#%%
class Moments(object):
    def __init__(self):
        return

    def fit(self, ims):
        features = []

        for im in ims:
            fs = self._internal_features(self._process_image(im))

            if fs.shape[0] > 0:
                features.append(fs)

        features = numpy.concatenate(features)

        self.bin_edges = []
        for i in range(features.shape[1]):
            _, bin_edges = numpy.histogram(features[:, i], 20)

            self.bin_edges.append(bin_edges)

        return self.bin_edges

    def get_features(self, ims):
        Xs = []

        for im in ims:
            xs0 = []
            fs = self._process_image(im)
            for k in range(0, im.shape[0] / b, 1):
                for j in range(0, im.shape[1] / b, 1):
                    xs = []
                    features = self._internal_features(fs, k * b, (k + 1) * b, j * b, (j + 1) * b)

                    if len(features) > 0:
                        for i in range(features.shape[1]):
                            xs.extend(numpy.histogram(features[:, i], bins = self.bin_edges[i])[0])

                        xs0.append(xs)

            if len(xs0) > 0:
                Xs.append(xs0)

        return numpy.array(Xs)

    def _process_image(self, im):
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

            features.append(((xm, ym), [numpy.sqrt(m00), w1, w2, u20, u02]))#

        return features

    def _internal_features(self, features, imin = None, imax = None, jmin = None, jmax = None):
        if imin == None:
            imin = 0

        if imax == None:
            imax = im.shape[0]

        if jmin == None:
            jmin = 0

        if jmax == None:
            jmax = im.shape[1]

        fs = []
        for (x, y), f in features:
            if x >= imin and x <= imax and y >= jmin and y <= jmax:
                fs.append(f)

        return numpy.array(fs)

class NN(object):
    def __init__(self):
        pass

    def fit(self, ims):
        tf.reset_default_graph()

        self.sess = tf.Session()

        self.tens = tf.placeholder(tf.float32, shape = [1, 256, 256, 3])

        # Create an instance, passing in the input data
        with tf.variable_scope("image_filters", reuse = False):
            self.net = googlenet.GoogleNet({'data' : self.tens})

        with tf.variable_scope("image_filters", reuse = True):
            self.net.load('/home/bbales2/caffe-tensorflow/googlenet.tf', self.sess, ignore_missing = True)

        self.target = [self.net.layers[name] for name in self.net.layers if name == 'inception_3a_1x1'][0]

        return

    def get_features(self, ims):
        hists = []
        for im in ims:
            im = numpy.pad(im, ((0, 256 - im.shape[0]), (0, 256 - im.shape[1])), 'constant')

            im2 = numpy.array((im, im, im)).astype('float')

            im2 -= im2.min()
            im2 /= im2.max() / 255.0

            im2 = numpy.rollaxis(im2, 1, 0)
            im2 = numpy.rollaxis(im2, 2, 1)

            mean = numpy.array([104., 117., 124.])

            for c in range(3):
                im2[:, :, c] -= mean[c]

            im2 = im2.reshape((1, im2.shape[0], im2.shape[1], im2.shape[2]))

            hist = self.sess.run(self.target, feed_dict = { self.tens : im2 })[0]

            hist = numpy.sum(hist[:b / 8, :b / 8].reshape((-1, hist.shape[-1])), axis = 0)
            #hist = microstructure.features.hists2boxes(hist, (b + 15) / 16, padding_mode = 'reflect')
            #print hist.shape

            hists.append(hist.reshape((-1, hist.shape[-1])))

        return hists

class LBP(object):
    def fit(self, ims):
        return

    def get_features(self, ims):
        ff = 7
        hists = []

        for im in ims:
            features = skimage.feature.local_binary_pattern(im, ff, 3.0)
            hist = microstructure.features.labels2boxes(features, int(2**ff), size = b + 1, stride = b, padding_mode = 'reflect')

            hist = hist[::2, ::2]

            hists.append(hist.reshape((-1, hist.shape[-1])))

        return hists

class HOG(object):
    def fit(self, ims):
        return

    def get_features(self, ims):

        hists = []
        for im in ims:
            hog = microstructure.features.hog2(im, bins = 20, stride = b, sigma = 1.0)
            hist = microstructure.features.hog2boxes(hog, b, padding_mode = 'reflect')

            hist = hist[::2, ::2]

            hists.append(hist.reshape((-1, hist.shape[-1])))

        return hists
