#!/usr/bin/python3

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

#print(digits.data)

#print(digits.target)

# Display an image of a number
#print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100)

#print(len(digits.data))

x, y = digits.data[:-10], digits.target[:-10]
clf.fit(x, y)

print('Prediction:', clf.predict(digits.data[[-3]]))

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
