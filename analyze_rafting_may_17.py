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
import sklearn.svm

os.chdir('/home/bbales2/microstructure_python')

import microstructure.features

os.chdir('/home/bbales2/lukestuff')

#%%

def get_features(im):
    b = 8
    features = skimage.feature.local_binary_pattern(im, 4, 3.0)
    hist = microstructure.features.labels2boxes(features, int(2**4), size = b + 1, stride = b, padding_mode = 'reflect')
    hist = microstructure.features.hists2boxes(hist, b = 9, padding_mode = 'reflect')

    return hist

filenames = [('scaled_rafting/ReneN5_0pct_10000x_Transverse_DendriteCore.TIF.png', 0.0),
         ('scaled_rafting/ReneN5_2pct_10000x_Transverse_DendriteCore.TIF.png', 2.0),
         ('scaled_rafting/ReneN5_5pct_10000x_Transverse_DendriteCore.TIF.png', 5.0)]

trainingx = []
trainingy = []

testingx = []
testingy = []

ims = []
trains = []
tests = []

for filename, y in filenames:
    im = skimage.io.imread(filename, as_grey = True).astype('double')
    im -= numpy.mean(im.flatten())
    im /= numpy.std(im.flatten())

    print im.shape

    ims.append(im)

    train = im[:3 * im.shape[0] / 4, :]
    test = skimage.filters.gaussian_filter(im[3 * im.shape[0] / 4:, :], 1.0)#sklearn.cross_validation.train_test_split(list(im))

    train = numpy.array(train)
    test = numpy.array(test)

    trainx = get_features(train)
    testx = get_features(test)

    trains.append(train)
    tests.append(test)

    trainy = numpy.ones((trainx.shape[0], trainx.shape[1])) * y
    testy = numpy.ones((testx.shape[0], testx.shape[1])) * y

    trainingx.extend(trainx.reshape((-1, trainx.shape[-1])))
    trainingy.extend(trainy.flatten())

    testingx.extend(testx.reshape((-1, testx.shape[-1])))
    testingy.extend(testy.flatten())

trainingx = numpy.array(trainingx)
trainingy = numpy.array(trainingy)

testingx = numpy.array(testingx)
testingy = numpy.array(testingy)

print trainingx.shape
print trainingy.shape

print testingx.shape
print testingy.shape

idxs = range(len(trainingx))
numpy.random.shuffle(idxs)

trainingx = trainingx[idxs]
trainingy = trainingy[idxs]

plt.imshow(test)
plt.show()

plt.imshow(train)
plt.show()

#%%

svm = sklearn.svm.SVR()

Cs = numpy.logspace(0, 2, 8)
gammas = numpy.logspace(-6, -4, 10)

accuracies = numpy.zeros((8, 10))
for i, c in enumerate(Cs):
    for j, gamma in enumerate(gammas):
        svm = sklearn.svm.SVR(C = c, gamma = gamma)

        svm.fit(trainingx[:1000], trainingy[:1000])

        accuracies[i, j] = svm.score(testingx, testingy)
        print i, j

plt.imshow(accuracies, interpolation = 'NONE')
plt.xticks(range(10), ['{:2.2e}'.format(v) for v in gammas])
plt.yticks(range(8), ['{:2.1e}'.format(v) for v in Cs])
plt.gcf().set_size_inches((10, 10))
plt.colorbar()
plt.show()

#%%

svm = sklearn.svm.SVR(C = 5.0, gamma = 1e-5)
svm.fit(trainingx[:1000], trainingy[:1000])

print svm.score(testingx, testingy)

rafting = svm.predict(testingx.reshape((-1, testingx.shape[-1]))).reshape((3 * testx.shape[0], testx.shape[1]))
plt.imshow(rafting)
plt.show()

#%%

svm = sklearn.linear_model.LinearRegression()
svm.fit(trainingx[:1000], trainingy[:1000])

print svm.score(testingx, testingy)

rafting = svm.predict(testingx.reshape((-1, testingx.shape[-1]))).reshape((3 * testx.shape[0], testx.shape[1]))
plt.imshow(rafting)
plt.show()

scores = []
for i in range(0, testingx.shape[1], 1):
    fs2 = list(reversed(numpy.argsort(numpy.abs(svm.coef_))))

    fs = fs2[:i + 1]

    print i, len(fs)

    lr2 = sklearn.linear_model.LinearRegression()
    lr2.fit(trainingx[:1000, fs], trainingy[:1000])

    rafting = lr2.predict(testingx[:, fs]).reshape((3 * testx.shape[0], testx.shape[1]))
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