import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


class Bootstrap:
    """
    Class used to perform bootstrap of dataset provided.

    :param means: (numpy array) Array were each entry is the mean value of a single bootstrap difference of two datasets.
    :param std: (float) Standard deviation of the means array.
    """
    def __init__(self):
        self.means = None
        self.std = 0

    def calc_means(self, data, iterations=1000, samples=100, replacement=True):
        """
        This method creates a bootstrap sampling of the input data and calculates the mean value of this sampled data.
        The process is repeated an n number of times defined by iterations. The calculations are stored in self.means
        numpy array.
        :param data: (array) values to be sampled
        :param iterations: (int) number of times the bootstrapping will be repeated
        :param samples: (int) number of samples to take from data
        :param replacement: (bool) if True data can be repeated, otherwise if False data can't be repeated
        :return: None
        """
        self.means = np.empty(iterations)
        for i in range(iterations):
            self.means[i] = np.mean(np.random.choice(data, size=samples, replace=replacement))

    def calc_std(self):
        """
        This method calculates the standard deviation of the self.means values and updates it on self.std.
        :return: None
        """
        self.std = np.std(self.means)

    def plot_hist(self, legend='data'):
        """
        Auxiliary method to plot the histogram distribution of the bootstrapped data.
        :param legend: (str) String that defines a custom label for the histogram.
        :return: None
        """
        sns.histplot(self.means, label=legend)
        plt.title('Histogram of Distribution')
        plt.xlabel('Event')
        plt.ylabel('Count')
        legend_patch = mpatches.Patch(color='C0', label=legend)
        plt.legend(handles=[legend_patch])

    def __sub__(self, other):
        """
        Magic method to perform subtraction between two Bootstrap objects. This method calculates the difference between
        the means of two Bootstrap objects and returns a Bootstrap object with this calculated means.
        :param other: (Bootstrap) Bootstrap object to compare.
        :return: (Bootstrap) Bootstrap object with means equal to self.means - other.means
        """
        result = Bootstrap()
        result.means = self.means - other.means
        return result
