#%%

import skimage.io
import os
import pandas
import matplotlib.pyplot as plt
import numpy
import sklearn.cross_validation
import patsy
import sklearn.mixture
import skimage.feature

os.chdir('/home/bbales2/microstructure_python')

import microstructure.features
reload(microstructure.features)

os.chdir('/home/bbales2/lukestuff')

#%%
df = pandas.DataFrame.from_csv('scaled_2.csv')
#%%

def buildFeatures(data, b = 8):
    Xs = []
    for index, row in data.iterrows():
        im = skimage.io.imread(row['f'], as_grey = True)
        features = skimage.feature.local_binary_pattern(im, 7, 3.0)
        hist = microstructure.features.labels2boxes(features, int(2**7), size = b + 1, stride = b, padding_mode = 'reflect')
        hist = microstructure.features.hists2boxes(hist, b = 9, padding_mode = 'reflect')

        Xs.append(hist)

        print index

    Xs = numpy.array(Xs)

    return Xs

def buildLabels(data, feature):
    labels = []

    for index, row in data.iterrows():
        labels.append(class2idx[row[feature]])

    labels = numpy.array(labels)

    return labels

#Y, X = patsy.dmatrices("made ~ home + season + type", train)
#test_ys, test_xs = patsy.dmatrices("made ~ home + season + type", test)
#%%
train, test = sklearn.cross_validation.train_test_split(df)

Xs = buildFeatures(train).astype('double')

#mean = numpy.mean(Xs, axis = (0, 1, 2))

#Xs -= mean

#std = numpy.std(Xs, axis = (0, 1, 2))
#Xs /= std

Xs2 = buildFeatures(test).astype('double')

#Xs2 -= mean
#Xs2 /= std
#%%

for feature in ["aged", "temp", "core", "pct", "type"]:
    classes = list(set(df[feature]))
    class2idx = dict([(v, k) for k, v in enumerate(classes)])

    labels = buildLabels(train, feature)
    labels2 = buildLabels(test, feature)

    Xs3 = []
    labels3 = []
    for i, (X, label) in enumerate(zip(Xs, labels)):
        Xs3.extend(X.reshape((-1, X.shape[-1])))
        labels3.extend(X.shape[0] * X.shape[1] * [label])

        #print i

    Xs3 = numpy.array(Xs3)
    labels3 = numpy.array(labels3)

    Xs5 = []
    labels5 = []
    for i, (X, label) in enumerate(zip(Xs2, labels2)):
        Xs5.extend(X.reshape((-1, X.shape[-1])))
        labels5.extend(X.shape[0] * X.shape[1] * [label])

        #print i

    Xs5 = numpy.array(Xs5)
    labels5 = numpy.array(labels5)

    idxs = range(Xs3.shape[0])
    numpy.random.shuffle(idxs)

    Xs4 = Xs3[idxs]
    labels4 = labels3[idxs]

    idxs = range(Xs5.shape[0])
    numpy.random.shuffle(idxs)

    Xs6 = Xs5[idxs]
    labels6 = labels5[idxs]

    svm = sklearn.svm.SVC(C = 1.0, gamma = 1e-5)
    print "{0}, Train: ".format(feature), numpy.mean(sklearn.cross_validation.cross_val_score(svm, Xs4[:10000], labels4[:10000]))
    svm.fit(Xs4[:10000], labels4[:10000])
    print "Test: ", 1.0 - numpy.count_nonzero(svm.predict(Xs6[:10000]) - labels6[:10000]) / 10000.0
    print ''
#%%
if True:

    Xs5 = []
    labels5 = []
    for i, (X, label) in enumerate(zip(Xs2, labels2)):
        boxes = microstructure.features.hists2boxes(X, b = 9, padding_mode = 'reflect')

        Xs5.extend(boxes.reshape((-1, boxes.shape[-1])))
        labels5.extend(boxes.shape[0] * boxes.shape[1] * [label])

        print i

    Xs5 = numpy.array(Xs5)
    labels5 = numpy.array(labels5)
#%%
Xs3 = numpy.array(Xs3)
labels3 = numpy.array(labels3)

idxs = range(Xs3.shape[0])
numpy.random.shuffle(idxs)

Xs4 = Xs3[idxs]
labels4 = labels3[idxs]
#%%
svm = sklearn.svm.SVC()

Cs = numpy.logspace(-2, 2, 8)
gammas = numpy.logspace(-5, 2, 10)

accuracies = numpy.zeros((8, 10))
for i, c in enumerate(Cs):
    for j, gamma in enumerate(gammas):
        svm = sklearn.svm.SVC(C = c, gamma = gamma)

        accuracies[i, j] = numpy.mean(sklearn.cross_validation.cross_val_score(svm, Xs4[:1000], labels4[:1000]))
        print i, j

plt.imshow(accuracies, interpolation = 'NONE')
plt.xticks(range(10), ['{:2.2e}'.format(v) for v in gammas])
plt.yticks(range(8), ['{:2.1e}'.format(v) for v in Cs])
plt.gcf().set_size_inches((10, 10))
plt.colorbar()
plt.show()
#%%
svm = sklearn.svm.SVC(C = 7.0, gamma = 1e-2)
print sklearn.cross_validation.cross_val_score(svm, Xs4[:10000], labels4[:10000])
svm.fit(Xs4[:10000], labels4[:10000])
#%%
numpy.count_nonzero(svm.predict(Xs3[0:15000]) - labels3[0:15000])
#%%

for feature in ["aged", "temp", "core", "pct", "type"]:
    classes = list(set(df[feature]))
    class2idx = dict([(v, k) for k, v in enumerate(classes)])

    #labels = buildLabels(train, feature)
    #labels2 = buildLabels(test, feature)

    #Xs3 = []
    #labels3 = []
    for i, (X, label) in enumerate(zip(Xs, labels)):
        boxes = microstructure.features.hists2boxes(X, b = 9, padding_mode = 'reflect')

        toplot = svm.predict(boxes.reshape((-1, boxes.shape[-1]))).reshape((boxes.shape[0], boxes.shape[1]))
        #labels3.extend(boxes.shape[0] * boxes.shape[1] * [label])

        row = train.iloc[i]

        im = skimage.io.imread(row['f'], as_grey = True)
        plt.imshow(im, interpolation = 'NONE', cmap = plt.cm.gray)
        plt.imshow(toplot, alpha = 0.5, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0))
        plt.show()

        print "{0}: {1}: {2}".format(i, feature, row[feature])

    1/0
    import sklearn.svm

    svm = sklearn.svm.SVC()

    print "{0}, Train: ".format(feature), numpy.mean(sklearn.cross_validation.cross_val_score(svm, Xs, labels))
    svm.fit(Xs, labels)
    print "Test: ", 1.0 - numpy.count_nonzero(svm.predict(Xs2) - labels2) / float(len(labels2))

#%%

gmm = sklearn.mixture.GMM(20)
sample = Xs.reshape((-1, Xs.shape[-1]))
numpy.random.shuffle(sample)

gmm.fit(sample[:10000])
#%%
hog = microstructure.features.HOG(8)

Xs3 = []
labels3 = []

for i, (index, row) in enumerate(df.iterrows()):#df.loc[87:88]
    im = skimage.io.imread(row['f'], as_grey = True)

    im = im[:(im.shape[0] / 8) * 8, :(im.shape[1] / 8) * 8]
    hogs = hog.run(im)

    hogs -= mean
    hogs /= std

    labels = gmm.predict(hogs.reshape((-1, hogs.shape[-1]))).reshape((hogs.shape[0], hogs.shape[1]))

    boxes = labels2boxes(labels, 20, b = 15, padding_mode = 'reflect')

    Xs3.extend(boxes.reshape((-1, boxes.shape[-1])))
    labels3.extend(boxes.shape[0] * boxes.shape[1] * [class2idx[row[feature]]])
    #plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
    #plt.imshow(labels, vmin = 0.0, vmax = 20.0, alpha = 0.5, extent = (0, im.shape[1], im.shape[0], 0), interpolation = 'NONE')
    #plt.show()

    #1/0

    print i, '/', len(df)
gmm.predict_proba(sample[:5])


#%%
Xs3 = numpy.array(Xs3)
labels3 = numpy.array(labels3)

import sklearn.svm

idxs = range(len(Xs3))
numpy.random.shuffle(idxs)
Xs4 = Xs3[idxs]
labels4 = labels3[idxs]
#%%
accuracies = numpy.zeros((8, 10))
for i, c in enumerate(numpy.logspace(-2, 2, 8)):
    for j, gamma in enumerate(numpy.logspace(-5, 0, 10)):
        svm = sklearn.svm.SVC(C = c, gamma = gamma)

        accuracies[i, j] = numpy.mean(sklearn.cross_validation.cross_val_score(svm, Xs4[:1000], labels4[:1000]))
        print i, j

plt.imshow(accuracies, interpolation = 'NONE')
plt.xticks(range(10), ['{:2.2f}'.format(v) for v in numpy.linspace(-5, 0, 10)])
plt.yticks(range(8), ['{:2.2f}'.format(v) for v in numpy.linspace(-2, 2, 8)])
plt.colorbar()
plt.show()
#%%
svm = sklearn.svm.SVC(C = 1.0, gamma = numpy.power(10.0, -4.0))#C = numpy.power(10.0, 1.0)

print numpy.mean(sklearn.cross_validation.cross_val_score(svm, Xs4[:10000], labels4[:10000]))
svm.fit(Xs4[:10000], labels4[:10000])
#%%
import sklearn.linear_model

sgdsvm = sklearn.linear_model.SGDClassifier()

#print numpy.mean(sklearn.cross_validation.cross_val_score(svm, Xs3[idxs], labels3[idxs]))
idxs = range(numpy.random.randint(0, len(Xs3), 10000)
sgdsvm.fit(Xs3[idxs], labels3[idxs])
#%%
out = svm.predict(Xs3[0:10000]) - labels3[0:10000]
print numpy.count_nonzero(out)
#%%

for i, (index, row) in enumerate(df.iterrows()):#df.loc[87:88]
    im = skimage.io.imread(row['f'], as_grey = True)

    im = im[:(im.shape[0] / 8) * 8, :(im.shape[1] / 8) * 8]
    hogs = hog.run(im)

    hogs -= mean
    hogs /= std

    labels = gmm.predict(hogs.reshape((-1, hogs.shape[-1]))).reshape((hogs.shape[0], hogs.shape[1]))

    boxes = labels2boxes(labels, 20, b = 15, padding_mode = 'reflect')

    labels = svm.predict(boxes.reshape((-1, boxes.shape[-1]))).reshape((boxes.shape[0], boxes.shape[1]))

    plt.imshow(im, interpolation = 'NONE', cmap = plt.cm.gray)
    plt.imshow(labels, alpha = 0.5, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0))
    plt.show()
    print "{0}: {1}: {2}".format((i, index), feature, row[feature])
