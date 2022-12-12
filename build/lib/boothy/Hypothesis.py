from .Bootstraping import Bootstrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


class Hypothesis(Bootstrap):
    """
    Class for calculating the p-value on a hypothesis test of two datasets by performing bootstrapping on both datasets.
    This class includes auxiliary methods that allows an easy understanding of the type of test performed. When using
    this class both datasets doesn't need to be of same size.

    :param bootstrap_diff: (Bootstrap) Bootstrap object with the calculated mean difference of objects bootstrap_one and
                            bootstrap_two
    :param data_one: (numpy array) Dataset used for bootstrap_one object
    :param data_two: (numpy array) Dataset used for bootstrap_two object
    :param bootstrap_one: (Bootstrap) Bootstrap object obtained using data_one
    :param bootstrap_two: (Bootstrap) Bootstrap object obtained using data_two
    :param mu: (float): Mean difference value between data_one mean and data_two mean
    :param null: (numpy array) Normal distribution with mean zero and standard deviation from the bootstrap difference
    :param p_val: (float) p-value of the null hypothesis
    :return: None
    """

    def __init__(self, data_one, data_two):
        super().__init__()
        self.bootstrap_diff = Bootstrap()
        self.data_one = data_one
        self.data_two = data_two
        self.bootstrap_one = Bootstrap()
        self.bootstrap_two = Bootstrap()
        self.mu = np.mean(data_one) - np.mean(data_two)
        self.null = None
        self.p_val = None

    def calc_bootstrap(self, iterations=1000, samples=100, replacement=True):
        """
        This method creates a bootstrap sampling of two input data and calculates the mean value of these sampled data.
        The process is repeated an n number of times defined by iterations. The calculations are stored in
        self.bootstrap_one.means and self.bootstrap_two.means. Calculation of standard deviations for each bootstrap is
        also obtained and stored as a parameter for each object.
        :param iterations: (int) number of times the bootstrapping will be repeated
        :param samples: (int) number of samples to take from data
        :param replacement: (bool) if True data can be repeated, otherwise if False data can't be repeated
        :return: None
        """
        self.bootstrap_one.calc_means(self.data_one, iterations, samples, replacement)
        self.bootstrap_one.calc_std()
        self.bootstrap_two.calc_means(self.data_two, iterations, samples, replacement)
        self.bootstrap_two.calc_std()

    def calc_diff(self):
        """
        This method calculates the difference of means and standard deviation between two bootstrapped objects.
        :return: None
        """
        self.bootstrap_diff = self.bootstrap_one - self.bootstrap_two
        self.bootstrap_diff.calc_std()

    def calc_null(self, iterations=1000):
        """
        This method calculates the null hypothesis data. For this a normal distribution with mean zero and standard
        deviation from the bootstrap difference is used.
        :param iterations: (int) number of times the bootstrapping will be repeated
        :return: None
        """
        self.null = np.random.normal(0.0, self.bootstrap_diff.std, iterations)

    def calc_p(self, null_type='>='):
        """
        This method will evaluate the null hypothesis. The null hypothesis to be evaluated depend on the
        null_type string.
        :param null_type: (str) The null type to be performed can be '>=', '<=' or '=='.
        :return: (int) p-value of the null hypothesis
        """
        if null_type == '>=':
            self.p_val = np.mean(self.null < self.mu)
        elif null_type == '<=':
            self.p_val = np.mean(self.null > self.mu)
        elif null_type == '==':
            self.p_val = np.mean(self.null <= -self.mu) + np.mean(self.null >= self.mu)
        else:
            self.p_val = None
            raise "Error: hypothesis null_type must be '>=', '<=' or '=='"
        return self.p_val

    def eval(self, iterations=1000, samples=100, replacement=True, null_type='>='):
        """
        Method that will perform the bootstrapping process and evaluate the null hypothesis.
        :param iterations: (int) number of times the bootstrapping will be repeated
        :param samples: (int) number of samples to take from data
        :param replacement: (bool) if True data can be repeated, otherwise if False data can't be repeated
        :param null_type: (str) The null type to be performed can be '>=', '<=' or '=='.
        :return: (int) p-value of the null hypothesis
        """
        self.calc_bootstrap(iterations, samples, replacement)
        self.calc_diff()
        self.calc_null(iterations)
        p_val = self.calc_p(null_type=null_type)
        return p_val

    def plot_hist_samples(self, legend_one='data one', legend_two='data two'):
        """
        Auxiliary method that plots the histogram distribution of two bootstrapped arrays.
        :param legend_one: (str) String that defines a custom label for the first histogram.
        :param legend_two: (str) String that defines a custom label for the second histogram.
        :return: None
        """
        self.bootstrap_one.plot_hist()
        self.bootstrap_two.plot_hist()
        legend_one_patch = mpatches.Patch(color='C0', label=legend_one)
        legend_two_patch = mpatches.Patch(color='C1', label=legend_two)
        plt.legend(handles=[legend_one_patch, legend_two_patch])
        plt.title('Distribution of Bootstraping Samples')
        # TODO: Fix problem with duplicated legends
        # TODO: Check this https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend

    def plot_hist_diff(self, legend_null='null', legend_alt='alternative', null_type='>=', num_stds=5):
        """
        Auxiliary method that plots the histogram distribution of the null and alternative hypothesis. This plot also
        includes the hypothesis testing region to be used to evaluate p (shadowed in orange) and the threshold regions
        (with dashed red lines).
        :param legend_null: (str) String that defines a custom label for the null hypothesis histogram.
        :param legend_alt: (str) String that defines a custom label for the alternative hypothesis histogram.
        :param null_type: (str) The null type to be performed can be '>=', '<=' or '=='.
        :param num_stds: (int) Number of standard deviations to be used a left or right limit for plotting the
                        hypothesis testing region.
        :return: None
        """
        sns.histplot(self.null, label=legend_null)
        self.bootstrap_diff.plot_hist(legend_alt)
        legend_null_patch = mpatches.Patch(color='C0', label=legend_null)
        legend_alt_patch = mpatches.Patch(color='C1', label=legend_alt)
        delta = num_stds * np.max([self.bootstrap_diff.std, np.std(self.null)])
        if null_type == '>=':
            plt.axvspan(self.mu - delta, self.mu, alpha=0.25, color='orange')
            plt.axvline(x=self.mu, color='red', linestyle='--')
        elif null_type == '<=':
            plt.axvspan(self.mu, self.mu + delta, alpha=0.25, color='orange')
            plt.axvline(x=self.mu, color='red', linestyle='--')
        elif null_type == '==':
            plt.axvspan(-self.mu - delta, -self.mu, alpha=0.25, color='orange')
            plt.axvspan(self.mu, self.mu + delta, alpha=0.25, color='orange')
            plt.axvline(x=-self.mu, color='red', linestyle='--')
            plt.axvline(x=self.mu, color='red', linestyle='--')
        plt.legend(handles=[legend_null_patch, legend_alt_patch])
        plt.title('Distribution of Null and Alternative Bootstraping')
