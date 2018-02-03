#!/usr/bin/python3

def classify(features_train, labels_train):

    # Import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    # Create a classifier
    clf = GaussianNB()
    
    # Fit the classifier on the training features and labels
    pred = clf.fit(features_train, labels_train)
    
    # Return the fit classifier
    return pred
