import unittest
import matplotlib.pyplot as plt
from boothy import Bootstrap
import numpy as np


class TestBootstrapClass(unittest.TestCase):
    def test_calc_means(self):
        np.random.seed(42)
        bootstrap_means = np.loadtxt('bootstrap_one.txt')
        data = np.loadtxt('bootstrap_one_data.txt')
        bootstrap = Bootstrap()
        bootstrap.calc_means(data)
        self.assertListEqual(list(bootstrap.means), list(bootstrap_means),
                             'incorrect calc_means')

    def test_calc_std(self):
        np.random.seed(42)
        data = np.loadtxt('bootstrap_one_data.txt')
        bootstrap = Bootstrap()
        bootstrap.calc_means(data)
        bootstrap.calc_std()
        self.assertEqual(np.round(bootstrap.std, 2), 2.93, 'incorrect calc_std')

    def test_sub(self):
        np.random.seed(42)
        bootstrap_one = Bootstrap()
        bootstrap_one.calc_means(np.loadtxt('bootstrap_one_data.txt'))
        np.random.seed(42)  # We reset the seed again
        bootstrap_two = Bootstrap()
        bootstrap_two.calc_means(np.loadtxt('bootstrap_two_data.txt'))
        bootstrap_diff = bootstrap_one - bootstrap_two
        boostrap_diff_data = np.loadtxt('bootstrap_diff.txt')

        self.assertListEqual(list(bootstrap_one.means), list(np.loadtxt('bootstrap_one.txt')),
                             'incorrect bootstrap one')
        self.assertListEqual(list(bootstrap_two.means), list(np.loadtxt('bootstrap_two.txt')),
                             'incorrect bootstrap two')
        self.assertListEqual(list(bootstrap_diff.means), list(boostrap_diff_data),
                             'incorrect bootstrap difference')

    def test_plot_hist(self):
        np.random.seed(42)
        data = np.loadtxt('bootstrap_one_data.txt')
        bootstrap = Bootstrap()
        bootstrap.calc_means(data)
        bootstrap.plot_hist()
        plt.show()

    # TODO: Make Hypothesis unittests

    if __name__ == '__main__':
        unittest.main()
