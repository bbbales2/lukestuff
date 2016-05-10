#%%

import skimage.io
import os
import pandas
import matplotlib.pyplot as plt
import numpy
import sklearn.cross_validation
import patsy
import sklearn.mixture

os.chdir('/home/bbales2/microstructure_python')

import microstructure.features
reload(microstructure.features)

os.chdir('/home/bbales2/lukestuff')

#%%
df = pandas.DataFrame.from_csv('scaled_2.csv')
#%%

def buildFeatures(data):
    Xs = []
    hog = microstructure.features.HOG(8)
    for index, row in data.iterrows():#df.loc[87:88]
        im = skimage.io.imread(row['f'], as_grey = True)
        hogs = hog.run(im)

        hogs = hogs.reshape((-1, hogs.shape[-1]))

        Xs.append(hogs)

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
trained = {}
tested = {}
for i in range(10):
    train, test = sklearn.cross_validation.train_test_split(df)

    Xs = buildFeatures(train)

    mean = numpy.mean(Xs, axis = (0, 1))

    Xs -= mean

    std = numpy.std(Xs, axis = (0, 1))
    Xs /= std

    Xs2 = buildFeatures(test)

    Xs2 -= mean
    Xs2 /= std

    for feature in ["aged", "temp", "core", "pct", "type"]:
        if feature not in trained:
            trained[feature] = []
            tested[feature] = []

        classes = list(set(df[feature]))
        class2idx = dict([(v, k) for k, v in enumerate(classes)])

        labels = buildLabels(train, feature)
        labels2 = buildLabels(test, feature)

        gmms = {}

        for c in classes:
            gmm = sklearn.mixture.GMM(20)
            sample = numpy.concatenate(Xs[labels == class2idx[c]])
            idxs = numpy.random.randint(0, len(sample), 10000)
            sample = sample[idxs]

            gmm.fit(sample)

            gmms[c] = gmm

            #print gmm.means_

        results = []

        for X, label in zip(Xs, labels):
            scores = []
            for c, gmm in gmms.items():
                scores.append((c, sum(gmm.score(X))))

            scores = sorted(scores, key = lambda x : x[1])

            #print scores, classes[label], scores[-1][0]

            results.append(classes[label] == scores[-1][0])

        print feature

        print 'Training: ', numpy.mean(results)
        trained[feature].append(numpy.mean(results))

        results = []
        for X, label in zip(Xs2, labels2):
            scores = []
            for c, gmm in gmms.items():
                scores.append((c, sum(gmm.score(X))))

            scores = sorted(scores, key = lambda x : x[1])

            #print scores, classes[label], scores[-1][0]

            results.append(classes[label] == scores[-1][0])

        print 'Testing: ', numpy.mean(results)
        tested[feature].append(numpy.mean(results))

print 'Property | Train/test | mean(correct) | std(mean(correct))'
for c in trained:
    print "{0} | train | {1:0.2f} | {2:0.3f}".format(c, numpy.mean(trained[c]), numpy.std(trained[c]))
    print "{0} | test  | {1:0.2f} | {2:0.3f}".format(c, numpy.mean(tested[c]), numpy.std(tested[c]))

for c in trained:
    print list(set(df[c]))
#%%
import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()
lr.fit(Xs, labels)
#%%
totalXs = {}
for idx in Xs:
X12 = numpy.array(numpy.concatenate(Xs))
X22 = numpy.array(X2)

X12 -= numpy.mean(X12, axis = 0)
X12 /= numpy.std(X12, axis = 0)

X22 -= numpy.mean(X22, axis = 0)
X22 /= numpy.std(X22, axis = 0)

import sklearn.mixture

gmm = sklearn.mixture.GMM(10)
gmm.fit(X22)

#%%
import sklearn.decomposition

pca = sklearn.decomposition.PCA(3)

pca.fit(X1 + X2)

out1 = pca.transform(X1)
out2 = pca.transform(X2)

plt.plot(out1[:, 1], out1[:, 2], 'r.')
plt.plot(out2[:, 1], out2[:, 2], 'b.')
plt.show()
#print gmm.means_
##%%
print sum(gmm.score(X12))
print sum(gmm.score(X22))
#%%
#plt.hist(X12[:, 11], bins = 1000)
#plt.show()
#%%
"aged"
"temp"
"core"
"pct"
"type"
#%%