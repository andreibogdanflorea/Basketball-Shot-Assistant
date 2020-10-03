import numpy
from scipy.spatial import ConvexHull
from sklearn.svm import SVC
import plotData
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as pyplot

def featureNormalization(X):
    mu = X.mean(axis = 0)
    std = X.std(axis = 0)
    X_norm = (X - mu) / std

    return (X_norm, mu, std)


def shiftConvexHull(X, offset_ratio):
    hull = ConvexHull(X)
    hull_points = X[hull.vertices, :]
    new_hull_points = offset_ratio * hull_points
    points = new_hull_points

    for i in range(len(new_hull_points) - 1):
        a = new_hull_points[i, :]
        b = new_hull_points[i + 1, :]
        new_points = numpy.array(
            [9 / 10 * a + 1 / 10 * b, 8 / 10 * a + 2 / 10 * b, 7 / 10 * a + 3 / 10 * b, 6 / 10 * a + 4 / 10 * b,
             5 / 10 * a + 5 / 10 * b, 4 / 10 * a + 6 / 10 * b, 3 / 10 * a + 7 / 10 * b, 2 / 10 * a + 8 / 10 * b,
             1 / 10 * a + 9 / 10 * b])
        points = numpy.append(points, new_points, axis=0)

    a = new_hull_points[len(new_hull_points) - 1, :]
    b = new_hull_points[0, :]
    new_points = numpy.array(
        [9 / 10 * a + 1 / 10 * b, 8 / 10 * a + 2 / 10 * b, 7 / 10 * a + 3 / 10 * b, 6 / 10 * a + 4 / 10 * b,
         5 / 10 * a + 5 / 10 * b, 4 / 10 * a + 6 / 10 * b, 3 / 10 * a + 7 / 10 * b, 2 / 10 * a + 8 / 10 * b,
         1 / 10 * a + 9 / 10 * b])
    points = numpy.append(points, new_points, axis=0)

    return points


def findOptimalPoint(X, y, y_with_swishes, parameter_names, num_pts=None, y_initial=None, plot_contours=False):
    model = CalibratedClassifierCV(SVC(gamma='auto'), cv=5, method='sigmoid')
    classifier = model.fit(X, y)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = plotData.makeMeshGrid(X0, X1)
    Z = classifier.predict_proba(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    max_index = numpy.where(Z.ravel() == Z.ravel().max())
    optimal_point = numpy.c_[xx.ravel(), yy.ravel()][max_index][0]
    probability = classifier.predict_proba(optimal_point.reshape(1, -1))

    if plot_contours is True:
        fig, ax = plotData.plotPoints(X[:num_pts, :], y_initial, parameter_names, 'Colormap of shot probabilities at different angles')
        min_x = min(X[:num_pts, 0])
        max_x = max(X[:num_pts, 0])
        min_y = min(X[:num_pts, 1])
        max_y = max(X[:num_pts, 1])
        ax.set_xlim([min_x - 1, max_x + 1])
        ax.set_ylim([min_y - 1, max_y + 1])

        contour = ax.contourf(xx, yy, Z, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Probability of a made shot')
        ax.scatter(optimal_point[0], optimal_point[1], color='green', s=60, marker='x', label='Optimal Parameteres')
        ax.annotate("{:.2f}%".format(probability[0, 1] * 100), optimal_point, color='green', size=10, xytext=(optimal_point[0] - 0.4, optimal_point[1] + 0.3))
        pyplot.legend()
        return fig, ax

    return optimal_point, probability




