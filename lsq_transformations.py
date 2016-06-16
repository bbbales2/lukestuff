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

#%%
im = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(1, 0), as_grey = True)
im2 = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(1, 5), as_grey = True)
hog = microstructure.features.HOG(16, padding = False)
hogs = hog.run(im)

plt.imshow(im)
plt.show()

fs = hogs.reshape((-1, hogs.shape[-1]))

for i in range(fs.shape[1]):
    plt.hist(fs[:, i])
    plt.show()
#%%
def moment_feats(im):
    thresh = skimage.filters.threshold_otsu(im)

    #plt.imshow(im > thresh)
    #plt.show()

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
        #m12 = mahotas.moments(precipitates, 1, 2)
        #m21 = mahotas.moments(precipitates, 2, 1)
        #m22 = mahotas.moments(precipitates, 2, 2)

        #plt.imshow(precipitates)
        #plt.show()

        xm = m10 / m00
        ym = m01 / m00

        #if m00 < 1e-5:
        #    continue

        #u00 = m00
        u00 = 1.0
        u11 = m11 / m00 - xm * ym
        u20 = m20 / m00 - xm ** 2
        u02 = m02 / m00 - ym ** 2

        w1 = u00**2 / (2.0 * numpy.pi * (u20 + u02))
        w2 = u00**4 / (16.0 * numpy.pi * numpy.pi * (u20 * u02 - u11**2))

        if w1 < 0.0 or w1 > 1.0 or w2 < 0.0 or w2 > 1.0 or numpy.isnan(w1) or numpy.isnan(w2):
            continue
            print m00, m01, m10, m11, m02, m20
            print xm
            print ym
            print u00, u11, u20, u02
            print w1, w2
            1/0

        features.append([numpy.sqrt(m00)])#w1, w2, , u20, u02

    features = numpy.array(features)

    return features

out = moment_feats(im)
out2 = moment_feats(im2)
plt.hist(out, 100)
plt.show()
plt.hist(out2, 100)
plt.show()
#plt.plot(out[:, 2], out[:, 3], '*')
#plt.show()
#    1/0
#skimage.measure.moments(im > thresh, order=3)
#%%
im2 = im - numpy.mean(im.flatten())
im2 /= numpy.std(im2.flatten())

im2 += numpy.random.randn(im.shape[0], im.shape[1]) * 0.5
plt.imshow(im2, cmap = plt.cm.gray, interpolation = 'NONE')
plt.gcf().set_size_inches((10, 8))
plt.show()
#%%
sys.path.append('/home/bbales2/caffe-tensorflow')

import googlenet

import tensorflow as tf
sess = tf.InteractiveSession()

tens = tf.placeholder(tf.float32, shape = [1, 256, 256, 3])

# Create an instance, passing in the input data
with tf.variable_scope("image_filters", reuse = False):
    net = googlenet.GoogleNet({'data' : tens})

with tf.variable_scope("image_filters", reuse = True):
    net.load('/home/bbales2/caffe-tensorflow/googlenet.tf', sess, ignore_missing = True)
#%%
target = [net.layers[name] for name in net.layers if name == 'inception_4a_1x1'][0]

im = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(1, 0), as_grey = True)

im2 = numpy.array((im, im, im)).astype('float')

im2 -= im2.min()
im2 /= im2.max() / 255.0

im2 = numpy.rollaxis(im2, 1, 0)
im2 = numpy.rollaxis(im2, 2, 1)

mean = numpy.array([104., 117., 124.])

for c in range(3):
    im2[:, :, c] -= mean[c]

im2 = im2.reshape((1, im2.shape[0], im2.shape[1], im2.shape[2]))

hist = sess.run(target, feed_dict = { tens : im2 })[0]

print hist.shape
#%%
plt.plot(fs[:, 7], fs[:, 8], '*')
plt.show()
#%%
b = 9
features = skimage.feature.local_binary_pattern(im, 7, 3.0)
hist = microstructure.features.labels2boxes(features, int(2**7), size = b + 1, stride = b, padding_mode = 'reflect')
#%%

def nn_feats(im):
    im2 = numpy.array((im, im, im)).astype('float')

    im2 -= im2.min()
    im2 /= im2.max() / 255.0

    im2 = numpy.rollaxis(im2, 1, 0)
    im2 = numpy.rollaxis(im2, 2, 1)

    mean = numpy.array([104., 117., 124.])

    for c in range(3):
        im2[:, :, c] -= mean[c]

    im2 = im2.reshape((1, im2.shape[0], im2.shape[1], im2.shape[2]))

    hist = sess.run(target, feed_dict = { tens : im2 })[0]
    hist = microstructure.features.hists2boxes(hist, (b + 15) / 16, padding_mode = 'reflect')

    return hist.reshape((-1, hist.shape[-1]))

def lbp_feats(im):
    ff = 7
    features = skimage.feature.local_binary_pattern(im, ff, 3.0)
    hist = microstructure.features.labels2boxes(features, int(2**ff), size = b + 1, stride = b, padding_mode = 'reflect')

    return hist.reshape((-1, hist.shape[-1]))

def hog_feats(im):
    hog = microstructure.features.hog2(im, bins = 20, stride = b, sigma = 1.0)
    hist = microstructure.features.hog2boxes(hog, b, padding_mode = 'reflect')

    return hist.reshape((-1, hist.shape[-1]))

def run_test():
    trainFilenames = []
    testFilenames = []

    dups = 8
    for y in train:
        for r in range(dups):##rafting2a1h5_rotatedrafting2a1h5
            trainFilenames.append(('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(y, r * 4), float(y)))

    for y in test:
        for r in range(dups):#nrafting2anrafting2arafting2a1h5
            testFilenames.append(('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(y, 128 + r * 4), float(y)))

    def extract_features(filenames, sigma, noise):
        Xs = []
        Ys = []

        ims = []

        for filename, y in filenames:
            im = skimage.io.imread(filename, as_grey = True).astype('double')
            #print im.shape
            im -= numpy.mean(im.flatten())
            im /= numpy.std(im.flatten())

            for noiseSigma in numpy.linspace(*noise):
                if noise > 0.0:
                    im2 = im + numpy.random.randn(*im.shape) * noiseSigma

                if sigma > 0.0:
                    im2 = skimage.filters.gaussian(im2, sigma)

                ims.append(im2)

                trainx = get_features(im2)

                #Xs.append(trainx)
                Xs.extend(trainx)

                trainy = numpy.ones((trainx.shape[0])) * y
                #trainy = numpy.ones((trainx.shape[0], trainx.shape[1])) * y

                #print trainx.shape
                #print trainy.shape
                #Ys.append(trainy)
                Ys.extend(trainy)

        Xs = numpy.array(Xs)
        Ys = numpy.array(Ys)

        return Xs, Ys, numpy.array(ims)

    trainxs, trainys, trainims = extract_features(trainFilenames, sigma0, noise0)
    testxs, testys, testims = extract_features(testFilenames, sigma1, noise1)

    trainingx = trainxs.reshape((-1, trainxs.shape[-1]))
    trainingy = trainys.reshape((-1))
    idxs = range(len(trainingx))
    numpy.random.shuffle(idxs)

    trainingx = trainingx[idxs]
    trainingy = trainingy[idxs]

    svm = sklearn.linear_model.Ridge(1e-5)#LinearRegression#
    #print trainingx.shape, trainingy.shape, trainxs.shape, trainys.shape
    svm.fit(trainingx[:10000], trainingy[:10000])

    train_predicts = []
    train_truths = []

    train_predicts = svm.predict(trainingx)
    train_truths = trainingy.flatten()

    testingx = testxs.reshape((-1, testxs.shape[-1]))
    testingy = testys.reshape((-1))

    test_predicts = svm.predict(testingx)
    test_truths = testingy.flatten()

    return numpy.array(train_truths), numpy.array(train_predicts), numpy.array(test_truths), numpy.array(test_predicts)

    #train_errors = []
    for trainx, trainy, trainim in zip(trainxs, trainys, trainims):
        trainingx = trainx.reshape((-1, trainx.shape[-1]))
        trainingy = trainy.reshape((-1))

        #rms = numpy.sqrt(numpy.mean(svm.predict(trainingx)))

        #trainRMSs.append((rms, trainy[0, 0]))
        #trainRMSs.append((numpy.mean(svm.predict(trainingx)), trainy[0, 0]))
        train_predicts.extend(svm.predict(trainingx))
        train_truths.extend(trainy.flatten())

    #testRMSs = []
    test_predicts = []
    test_truths = []
    #test_errors = []
    for testx, testy, testim in zip(testxs, testys, testims):
        testingx = testx.reshape((-1, testx.shape[-1]))
        testingy = testy.reshape((-1))

        #print svm.score(testingx, testingy)
        #print numpy.mean((svm.predict(testingx) - testingy)**2)
        #rms = numpy.sqrt(numpy.mean(svm.predict(testingx)))
        #print numpy.sqrt(numpy.mean(svm.predict(testingx)))

        #test_errors.append((numpy.mean(svm.predict(testingx) - testy[0, 0])))
        test_predicts.extend(svm.predict(testingx))
        test_truths.extend(testy.flatten())

        #rafting = svm.predict(testingx).reshape((-1, testy.shape[-1]))
        #testims_merge = testims.reshape(-1, testims.shape[-1])
        #plt.imshow(testim, interpolation = 'NONE', cmap = plt.cm.gray)
        #plt.imshow(rafting, interpolation = 'NONE', alpha = 0.25, extent = (0, testim.shape[1], testim.shape[0], 0), vmin = 0.0, vmax = 9.0)
        #plt.gcf().set_size_inches((10, 10))
        #plt.colorbar()
        #plt.title(testy[0, 0])
        #plt.show()
    return numpy.array(train_truths), numpy.array(train_predicts), numpy.array(test_truths), numpy.array(test_predicts)
#%%
sigma0 = 0.0
sigma1 = 0.0
noise0 = [0.0, 0.0, 1]
noise1 = [0.0, 0.0, 1]

get_features = moment_feats

trainR2s2 = []
testR2s2 = []
for b in numpy.linspace(64, 96, 10):
    b = int(numpy.round(b))

    trains = [[0, 9], [0, 4, 9], [0, 3, 5, 7, 9], [4, 5], [4, 5, 6], [3, 4, 5, 6, 7]]
    tests = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 6, 8], [0, 1, 2, 3, 6, 7, 8, 9], [0, 1, 2, 3, 7, 8, 9], [0, 1, 2, 8, 9]]

    trainR2s = []
    testR2s = []
    for train, test in zip(trains, tests):
        trainxs, trainys, testxs, testys = run_test()

        positions = []
        values = []
        testOrTrain = []
        for xv in sorted(list(set(trainxs))):
            preds = trainys[trainxs == xv]
            values.append(preds)
            positions.append(xv)
            testOrTrain.append(0)

        tr2 = 1 - sum((trainxs - trainys)**2) / sum((trainys - numpy.mean(trainys))**2)
        print 'b: ', b
        sys.stdout.flush()

        print 'Train R^2 ', tr2
        print 'Train rms error per sample: ', numpy.sqrt(sum((trainxs - trainys)**2 / len(trainxs)))
        trainR2s.append(tr2)

        for xv in sorted(list(set(testxs))):
            preds = testys[testxs == xv]
            values.append(preds)
            positions.append(xv)
            testOrTrain.append(1)

        ttr2 = 1 - sum((testxs - testys)**2) / sum((testys - numpy.mean(testys))**2)

        print 'Test R^2 ', ttr2
        print 'Test rms error per sample: ', numpy.sqrt(sum((testxs - testys)**2 / len(testxs)))
        testR2s.append(ttr2)
        #continue
        #1/0
        positions, values, testOrTrain = zip(*sorted(zip(positions, values, testOrTrain), key = lambda x : x[0]))

        bp = plt.boxplot(values, positions = positions)#, labels = ['train'] * len(positions))
        plt.gca().set_ylim((-3.0, 13.0))
        plt.gca().set_xlim((-1.0, 10.0))
        # Boxplot styles stolen from http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
        boxColors = ['darkkhaki', 'royalblue']
        for i in range(len(positions)):
            box = bp['boxes'][i]

            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            # Alternate between Dark Khaki and Royal Blue
            k = testOrTrain[i]
            boxPolygon = matplotlib.patches.Polygon(boxCoords, facecolor = boxColors[k])
            plt.gca().add_patch(boxPolygon)

        # Finally, add a basic legend
        plt.figtext(0.135, 0.805, 'Training data', backgroundcolor = boxColors[0], color = 'black', weight='roman')
        plt.figtext(0.135, 0.855, 'Testing data', backgroundcolor = boxColors[1], color='white', weight='roman')
        plt.figtext(0.135, 0.755, 'Exact ref.', backgroundcolor = 'red', color='white', weight='roman')

        plt.xlabel('Truth')
        plt.ylabel('Prediction')

        plt.plot([0.0, 9.0], [0.0, 9.0], 'ro-')

        plt.show()

    trainR2s2.append(trainR2s)

    testR2s2.append(testR2s)
#%%
hogTestR2s2 = numpy.array(testR2s2)
hogTrainR2s2 = numpy.array(trainR2s2)

#%%
lbpTestR2s2 = numpy.array(testR2s2)
lbpTrainR2s2 = numpy.array(trainR2s2)

#%%
nnTestR2s2 = numpy.array(testR2s2)
nnTrainR2s2 = numpy.array(trainR2s2)

#%%
plt.plot(numpy.linspace(7, 96, 10), hogTestR2s2[:, 3:6], '--')
plt.plot(numpy.linspace(7, 96, 10), [0.0] * 10, 'k')
plt.plot(numpy.linspace(7, 96, 10), [1.0] * 10, 'k')
plt.title('HOG feature descriptor (nrafting2a), train noise (0, 1, 4), test noise 0.87')
plt.ylabel('R^2')
plt.xlabel('Diameter of feature descriptor')
plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'upper left')
plt.ylim(-2, 1.5)
plt.show()
plt.plot(numpy.linspace(7, 96, 10), lbpTestR2s2[:, 3:6], '--')
plt.plot(numpy.linspace(7, 96, 10), [0.0] * 10, 'k')
plt.plot(numpy.linspace(7, 96, 10), [1.0] * 10, 'k')
plt.title('LBP feature descriptor (nrafting2a), train noise (0, 1, 4), test noise 0.87')
plt.ylabel('R^2')
plt.xlabel('Diameter of feature descriptor')
plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'upper left')
plt.ylim(-2, 1.5)
plt.show()
plt.plot(numpy.linspace(7, 96, 10), nnTestR2s2[:, 3:6], '--')
plt.plot(numpy.linspace(7, 96, 10), [0.0] * 10, 'k')
plt.plot(numpy.linspace(7, 96, 10), [1.0] * 10, 'k')
plt.title('NN feature descriptor (nrafting2a), train noise (0, 1, 4), test noise 0.87')
plt.ylabel('R^2')
plt.xlabel('Diameter of feature descriptor')
plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'lower right')
plt.ylim(-2, 1.5)
plt.show()
#%%
    plt.plot(trainxs, trainys, 'rx')
    plt.plot(testxs, testys, 'bx')
    plt.title('Interpolation')
    plt.legend(['train', 'test'])
    plt.show()
#%%
scores = []
for i in range(0, testingx.shape[1], 1):
    fs2 = list(reversed(numpy.argsort(numpy.abs(svm.coef_))))

    fs = fs2[:i + 1]

    print i, len(fs)

    lr2 = sklearn.linear_model.LinearRegression()
    lr2.fit(trainingx[:1000, fs], trainingy[:1000])

    rafting = lr2.predict(testingx[:, fs]).reshape((-1, testys.shape[-1]))
    plt.imshow(rafting)
    plt.title(str(i))
    plt.show()

    scores.append(lr2.score(testingx[:, fs], testingy))
    #print

plt.plot(scores)
plt.show()
#%%
for i in fs2:
    print svm.coef_[i], "{0:04b}".format(i)

#%%

for im, (filename, y) in zip(ims, filenames):
#if True:
#    im = skimage.io.imread('gtdmix.png', as_grey = True).astype('double')
#    im -= numpy.mean(im.flatten())
#    im /= numpy.std(im.flatten())

    filename = 'gtdmix'

    print im.shape

    #im2, test = sklearn.cross_validation.train_test_split(list(im), test_size = 0.0)

    #im2 = numpy.array(im2)

    #print im2.shape

    features = get_features(im)

    rafting = svm.predict(features.reshape((-1, features.shape[-1]))).reshape((features.shape[0], features.shape[1]))

    #plt.imshow(im, interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.show()
    plt.imshow(im, interpolation = 'NONE', cmap = plt.cm.gray)
    plt.imshow(rafting, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0), alpha = 0.5, vmin = 0.0, vmax = 5.0)
    plt.title(filename)
    plt.gcf().set_size_inches((17, 10))
    plt.colorbar()
    plt.show()
    #plt.imshow(rafting, interpolation = 'NONE', vmin = 0.0, vmax = 5.0)
    #plt.show()
#%%

f0 = get_features(im)
f1 = get_features(im2)