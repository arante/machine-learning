from sklearn import tree

#
# Training data
#

# 0 = bumpy, 1 = smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 = apple, 1 = orange
labels = [0, 0, 1, 1]

#
# Training the classifier
#

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

#
# Making the predictions
#

print(clf.predict([[160, 0]]))
