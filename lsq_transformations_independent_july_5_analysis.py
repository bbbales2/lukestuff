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

import pickle

f = open('lsqdata')
data, features = pickle.load(f)
f.close()

#%%
trains = [[0, 9], [0, 4, 9], [0, 3, 5, 7, 9], [4, 5], [4, 5, 6], [3, 4, 5, 6, 7]]
tests = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 6, 8], [0, 1, 2, 3, 6, 7, 8, 9], [0, 1, 2, 3, 7, 8, 9], [0, 1, 2, 8, 9]]

#R2s = {}
trainR2s = {}
testR2s = {}
for ftype in features:
    fdata = features[ftype]
    for size in fdata:
        dsets = fdata[size]
        for dset in dsets:
            trainsd = dsets[dset]['train']
            testsd = dsets[dset]['test']

            #trainR2s[(ftype, size, dset)] = []
            #testR2s[(ftype, size, dset)] = []
            for idx, (train, test) in enumerate(zip(trains, tests)):
                trainXs = []
                trainYs = []
                testXs = []
                testYs = []

                for y in train:
                    trainXs.extend([v.flatten() for v in trainsd[y]])
                    trainYs.extend([y] * len(trainsd[y]))

                for y in test:
                    testXs.extend([v.flatten() for v in testsd[y]])
                    testYs.extend([y] * len(testsd[y]))

                trainXs = numpy.array(trainXs)
                trainYs = numpy.array(trainYs)
                testXs = numpy.array(testXs)
                testYs = numpy.array(testYs)

                svm = sklearn.linear_model.Ridge(1e-5)
                svm.fit(trainXs, trainYs)

                trainYsPredict = svm.predict(trainXs)
                testYsPredict = svm.predict(testXs)

                positions = []
                values = []
                testOrTrain = []
                for xv in sorted(list(set(trainYs))):
                    preds = trainYsPredict[trainYs == xv]
                    values.append(preds)
                    positions.append(xv)
                    testOrTrain.append(0)

                tr2 = 1 - sum((trainYsPredict - trainYs)**2) / sum((trainYs - numpy.mean(trainYs))**2)
                sys.stdout.flush()

                print 'Train R^2 ', tr2
                print 'Train rms error per sample: ', numpy.sqrt(sum((trainYsPredict - trainYs)**2 / len(trainYs)))
                trainR2s[(ftype, size, dset, idx)] = tr2

                for xv in sorted(list(set(testYs))):
                    preds = testYsPredict[testYs == xv]
                    values.append(preds)
                    positions.append(xv)
                    testOrTrain.append(1)

                ttr2 = 1 - sum((testYsPredict - testYs)**2) / sum((testYs - numpy.mean(testYs))**2)

                print 'Test R^2 ', ttr2
                print 'Test rms error per sample: ', numpy.sqrt(sum((testYsPredict - testYs)**2 / len(testYs)))

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
                plt.title('{0} fit, R = {1}, {2}'.format(ftype, size, 'coarsening' if dset == 'nrafting2a' else 'rafting'))

                plt.plot([0.0, 9.0], [0.0, 9.0], 'ro-')

                plt.show()

                testR2s[(ftype, size, dset, idx)] = ttr2
                #testR2s.append(tr2)
#%%

#%%
import pandas

df = pandas.DataFrame(dict(zip(['Algo', 'size', 'dset', 'testIdx'], zip(*testR2s))))

for idx, table in df.groupby(['dset', 'testIdx']):
    algos = []
    for idx2, table2 in table.groupby(['Algo']):
        if table2.Algo.iloc[0] == 'NN':
            continue

        data = []
        for ridx, row in table2.iterrows():
            data.append((row['size'], testR2s[(row.Algo, row['size'], row.dset, row.testIdx)]))

        data = sorted(data, key = lambda x : x[0])

        sizes, ys = zip(*data)
        algos.append(table2.Algo.iloc[0])
        plt.plot(sizes, ys, '--')
    plt.title('{0} for training idxs {1}'.format('coarsening' if table.dset.iloc[0] == 'nrafting2a' else 'rafting', trains[table.testIdx.iloc[0]]))
    plt.ylim([-1.5, 1.5])
    plt.xlabel('Feature detector window size (pixels)')
    plt.ylabel('Test R^2')
    plt.legend(algos, loc = 'bottom left')
    plt.plot(sizes, [1.0] * len(sizes), 'k')
    plt.plot(sizes, [0.0] * len(sizes), 'k')
    plt.show()
#sorted(testR2s, key = lambda x : (x[0], x[2], x[3], x[1]))
#%%
sizes = [32, 64, 128, 256]
dsets = ['nrafting2a', 'rafting2a1h5']

for ftype, size, dset in testR2s:
for ftype in R2s:
    fdata = R2s[ftype]
    for dset in dsets:
        for i, train in enumerate(trains):
            ys = []
            for size in sizes:
                ys.append(testR2s[ftype][size][dset]
            dsets = fdata[size]
            trainsd = dsets[dset]['train']
            testsd = dsets[dset]['test']

            for train, test in zip(trains, tests):


plt.plot(bs, hogTestR2s2[:, 3:6], '--')
plt.plot(bs, [0.0] * len(bs), 'k')
plt.plot(bs, [1.0] * len(bs), 'k')
plt.title('HOG feature descriptor (nrafting2a), train noise (0, 1, 4), test noise 0.87')
plt.ylabel('R^2')
plt.xlabel('Diameter of feature descriptor')
plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'bottom left')
plt.ylim(-2, 1.5)
plt.show()

#%%

def run_test():
    features = Features()

    def extract_features(filenames, sigma, noise, fit = False):
        Xs = []
        Ys = []
        ys = []

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
                ys.append(y)

        if fit:
            features.fit(ims)

        fs = features.get_features(ims)

        #print len(fs), len(ims)
        #print type(fs)
        #print fs.shape
        #print fs[0]
        #for l in fs[0]:
        #    print len(l)
        #print fs[0].shape

        for trainx, y in zip(fs, ys):
            #Xs.append(trainx)
            Xs.extend(trainx)

            trainy = numpy.ones(len(trainx)) * y
            Ys.extend(trainy)

        Xs = numpy.array(Xs)
        Ys = numpy.array(Ys)

        return Xs, Ys, numpy.array(ims)

    trainxs, trainys, trainims = extract_features(trainFilenames, sigma0, noise0, True)
    testxs, testys, testims = extract_features(testFilenames, sigma1, noise1)

    print trainxs.shape, trainys.shape
    print testxs.shape, testys.shape

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
noise0 = [0.0, 1.0, 4]
noise1 = [0.87, 0.87, 1.0]

results = []
for dset in ['nrafting2a', 'rafting2a1h5']:
    for Features in [Moments, NN, LBP, HOG]:
        bs = [16, 32, 48, 64]

        trainR2s2 = []
        testR2s2 = []
        for b in bs:#numpy.linspace(7, 96, 10):
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

        results.append((dset, Features, numpy.array(testR2s2), numpy.array(trainR2s2)))
#%%
for dset, Features, testR2s2, trainR2s2 in results:
    plt.plot(bs, testR2s2[:, 0:3], '--')
    plt.plot(bs, [0.0] * len(bs), 'k')
    plt.plot(bs, [1.0] * len(bs), 'k')
    Ftype = { NN : 'NN',
              Moments : 'Moments',
              HOG : 'HOG',
              LBP : 'LBP' }[Features]
    plt.title('{0} feature descriptor ({1}), train noise (0, 1, 4), test noise 0.87'.format(Ftype, dset))
    plt.ylabel('R^2')
    plt.xlabel('Diameter of feature descriptor')
    plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'bottom left')
    plt.ylim(-2, 1.5)
    plt.show()
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
plt.plot(bs, lbpTestR2s2[:, 3:6], '--')
plt.plot(bs, [0.0] * len(bs), 'k')
plt.plot(bs, [1.0] * len(bs), 'k')
plt.title('LBP feature descriptor (nrafting2a), train noise (0, 1, 4), test noise 0.87')
plt.ylabel('R^2')
plt.xlabel('Diameter of feature descriptor')
plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'bottom left')
plt.ylim(-2, 1.5)
plt.show()
plt.plot(bs, nnTestR2s2[:, 3:6], '--')
plt.plot(bs, [0.0] * len(bs), 'k')
plt.plot(bs, [1.0] * len(bs), 'k')
plt.title('NN feature descriptor (nrafting2a), train noise (0, 1, 4), test noise 0.87')
plt.ylabel('R^2')
plt.xlabel('Diameter of feature descriptor')
plt.legend(['Two train points', 'Three train points', 'Five train points', 'reference levels (0.0 - 1.0)'], loc = 'bottom left')
plt.ylim(-2, 1.5)
plt.show()
