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

os.chdir('/home/bbales2/microstructure_python')

import microstructure.features

os.chdir('/home/bbales2/lukestuff')

#%%
im = skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(1, 0), as_grey = True)
hog = microstructure.features.HOG(16, padding = False)
hogs = hog.run(im)

plt.imshow(im)
plt.show()

fs = hogs.reshape((-1, hogs.shape[-1]))

for i in range(fs.shape[1]):
    plt.hist(fs[:, i])
    plt.show()
#%%
plt.plot(fs[:, 7], fs[:, 8], '*')
plt.show()
#%%

def run_test():
    trainFilenames = []
    testFilenames = []

    def get_features(im):
        features = skimage.feature.local_binary_pattern(im, 7, 3.0)
        hist = microstructure.features.labels2boxes(features, int(2**7), size = b + 1, stride = b, padding_mode = 'reflect')
        #hog = microstructure.features.HOG(16, padding = False)
        #hist = hog.run(im)

        return hist

    dups = 8
    for y in train:
        for r in range(dups):##rafting2a1h5_rotated
            trainFilenames.append(('/home/bbales2/rafting/nrafting2a/images_{0}/signal{1}.png'.format(y, r * 4), float(y)))

    for y in test:
        for r in range(dups):#nrafting2a
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

            if noise > 0.0:
                im += numpy.random.randn(*im.shape) * noise

            if sigma > 0.0:
                im = skimage.filters.gaussian(im, sigma)

            ims.append(im)

            trainx = get_features(im)

            Xs.append(trainx)

            trainy = numpy.ones((trainx.shape[0], trainx.shape[1])) * y

            Ys.append(trainy)

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

    svm = sklearn.linear_model.LinearRegression()
    svm.fit(trainingx[:10000], trainingy[:10000])

    train_predicts = []
    train_truths = []
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
    for testx, testy, testim in zip(testxs, testys, testims):
        testingx = testx.reshape((-1, testx.shape[-1]))
        testingy = testy.reshape((-1))

        #print svm.score(testingx, testingy)
        #print numpy.mean((svm.predict(testingx) - testingy)**2)
        #rms = numpy.sqrt(numpy.mean(svm.predict(testingx)))
        #print numpy.sqrt(numpy.mean(svm.predict(testingx)))

        #testRMSs.append((numpy.mean(svm.predict(testingx)), testy[0, 0]))
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
noise0 = 0.0
noise1 = 0.0

b = 51

trains = [[0, 9], [0, 4, 9], [0, 3, 5, 7, 9], [4, 5], [4, 5, 6], [3, 4, 5, 6, 7]]
tests = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 6, 8], [0, 1, 2, 3, 6, 7, 8, 9], [0, 1, 2, 3, 7, 8, 9], [0, 1, 2, 8, 9]]

for train, test in zip(trains, tests):
    trainxs, trainys, testxs, testys = run_test()

    positions = []
    values = []
    testOrTrain = []
    for xv in sorted(list(set(trainxs))):
        values.append(trainys[trainxs == xv])
        positions.append(xv)
        testOrTrain.append(0)

    for xv in sorted(list(set(testxs))):
        values.append(testys[testxs == xv])
        positions.append(xv)
        testOrTrain.append(1)

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