import sys
import numpy
import plotData
import processData
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def loadDataset(filename):
    # Loading Data from file
    print('Loading Data...')
    data = numpy.loadtxt(open(filename, 'r'), delimiter=",", dtype=numpy.ndarray)
    parameter_names = data[0, :]
    X = numpy.array(data[1:, :2], dtype=float)
    y = numpy.array(data[1:, 2], dtype=float)

    return X, y, parameter_names


def visualizeDataset(X, y, parameter_names):
    # Visualization
    print('Visualizing the dataset of shots')
    fig, ax = plotData.plotPoints(X, y, parameter_names, 'Dataset of shots')
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


def visualizeDecisionBoundary(classifier, X, num_pts, y_with_swishes, parameter_names):
    print('Visualizing the decision boundary for predicting a made or missed shot')
    fig, ax = plotData.visualizeDecisionBoundary(classifier, X, num_pts, y_with_swishes, parameter_names)

    return fig, ax


def optimalParameters(X, y, y_with_swishes, parameter_names, mu, std):
    # Calculating the optimal release angles
    print('Calculating the optimal release angles')
    optimal_point, probability = processData.findOptimalPoint(X, y, y_with_swishes, parameter_names, plot_contours=False)
    optimal_parameters = optimal_point * std + mu

    return optimal_parameters, probability

def visualizeColormap(X, y, y_with_swishes, num_pts, y_initial, parameter_names):
    # Visualizing Colormap of Free Throw probabilities at different angles
    print('Visualizing Colormap of shot probability at different parameters')
    fig, ax = processData.findOptimalPoint(X, y, y_with_swishes, parameter_names, num_pts, y_initial, plot_contours=True)

    return fig, ax
