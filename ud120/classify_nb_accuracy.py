#!/usr/bin/python3

def accuracy(features_train, labels_train, features_test, labels_test):
    '''Compute the accuracy of your Gaussian Naive Bayes classifier.'''

    # Imports sklearn modules
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    # Create a classifier
    clf = GaussianNB
    
    # Fit the classifier on training features and labels
    clf.fit(features_train, labels_train)

    # Use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    
    # Calculate and test the accuracy
    accuracy = accuracy_score(labels_test, pred)
    
    return accuracy
