import numpy
import matplotlib.pyplot as pyplot
from matplotlib import colors

def plotPoints(X, y, parameter_names, title='Figure'):
    swishes = numpy.where(y == 2)
    makes = numpy.where(y == 1)
    misses = numpy.where(y == 0)

    fig, ax = pyplot.subplots(num=title)
    fig.set_size_inches(8, 6)
    pyplot.xlabel(parameter_names[0])
    pyplot.ylabel(parameter_names[1])

    if title is not 'Figure':
        pyplot.title(title)

    pyplot.scatter(X[misses, 0], X[misses, 1], color='red', marker='x', label='Misses')
    pyplot.scatter(X[makes, 0], X[makes, 1], color='black', marker='o', label='Makes')
    pyplot.scatter(X[swishes, 0], X[swishes, 1], color='blue', marker='o', label='Swishes')

    pyplot.legend(loc="upper left")

    return fig, ax


def makeMeshGrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

    return xx, yy


def plot_contours(ax, classifier, xx, yy, **params):
    Z = classifier.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)

    return out


def visualizeDecisionBoundary(classifier, X, num_pts, y_initial, parameter_names):
    title = ('Decision Boundary - shows angles that result in the highest shot percentage')
    fig, ax = plotPoints(X[:num_pts, :], y_initial, title)

    min_x = min(X[:num_pts, 0])
    max_x = max(X[:num_pts, 0])
    min_y = min(X[:num_pts, 1])
    max_y = max(X[:num_pts, 1])
    ax.set_xlim([min_x - 1, max_x + 1])
    ax.set_ylim([min_y - 1, max_y + 1])

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = makeMeshGrid(X0, X1)
    colormap = colors.ListedColormap(['w', 'k', 'b'])
    plot_contours(ax, classifier, xx, yy, cmap=colormap, alpha=0.3)

    ax.set_xlabel(parameter_names[0])
    ax.set_ylabel(parameter_names[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc='upper left')

    return fig, ax

