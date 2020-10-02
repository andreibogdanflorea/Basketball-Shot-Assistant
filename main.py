import sys
import numpy
import plotData
import processData
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def loadDataset(filename):
    # Loading Data from file
    print('Loading Data...')
    data = numpy.loadtxt(open(filename, 'r'), delimiter=",")
    X = data[:, :2]
    y = data[:, 2]

    return X, y


def visualizeDataset(X, y):
    # Visualization
    print('Visualizing the dataset of Free Throw shots')
    fig, ax = plotData.plotPoints(X, y, 'Dataset of Free throw shots')
    return fig, ax


def normalizeFeatures_addConvexHullPoints(X, y):
    # Feature normalization and addition of "reference" misses (at abnormal angles)
    print('Normalizing features and adding reference misses at abnormal angles')
    X_norm, mu, std = processData.featureNormalization(X)

    # offset ratio to shift the convex hull points from the centroid with
    offset_ratio = 1.5
    points_to_add = processData.shiftConvexHull(X_norm, offset_ratio)

    # add those points to the dataset as misses, they are "abnormal" and are most likely misses
    X_norm = numpy.append(X_norm, points_to_add, axis=0)
    y = numpy.append(y, numpy.zeros(len(points_to_add)), axis=0)
    y_with_swishes = y.copy()
    y[y == 2] = 1

    return X_norm, y, mu, std, y_with_swishes


def trainModel(X, y):
    # Training the model
    print('Training the model')
    model = CalibratedClassifierCV(SVC(gamma='auto'), cv=5, method='sigmoid')
    classifier = model.fit(X, y)

    return classifier


def visualizeDecisionBoundary(classifier, X, num_pts, y_with_swishes):
    print('Visualizing the decision boundary for predicting a made or missed Free Throw shot')
    fig, ax = plotData.visualizeDecisionBoundary(classifier, X, num_pts, y_with_swishes)

    return fig, ax


def optimalParameters(X, y, y_with_swishes, mu, std):
    # Calculating the optimal release angles
    print('Calculating the optimal release angles')
    optimal_point, probability = processData.findOptimalPoint(X, y, y_with_swishes, plot_contours=False)
    optimal_parameters = optimal_point * std + mu

    return optimal_parameters, probability

def visualizeColormap(X, y, y_with_swishes, num_pts, y_initial):
    # Visualizing Colormap of Free Throw probabilities at different angles
    print('Visualizing Colormap of Free Throw probabilities at different angles')
    fig, ax = processData.findOptimalPoint(X, y, y_with_swishes, num_pts, y_initial, plot_contours=True)

    return fig, ax
