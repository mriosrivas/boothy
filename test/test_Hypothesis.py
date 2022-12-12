import unittest
import numpy as np
from boothy import Hypothesis
import matplotlib.pyplot as plt

condition = '=='


class TestHypothesisClass(unittest.TestCase):
    def setUp(self):
        data_one = np.loadtxt('bootstrap_one_data.txt')
        data_two = np.loadtxt('bootstrap_two_data.txt')
        self.hypo = Hypothesis(data_one, data_two)

    def test_calc_bootstrap(self):
        self.hypo.calc_bootstrap()
        bootstrap_one_mean = np.mean(np.loadtxt('bootstrap_one.txt'))
        bootstrap_two_mean = np.mean(np.loadtxt('bootstrap_two.txt'))
        self.assertAlmostEqual(float(np.mean(self.hypo.bootstrap_one.means)), float(bootstrap_one_mean), 0,
                               'bootstrap one error')
        self.assertAlmostEqual(float(np.mean(self.hypo.bootstrap_two.means)), float(bootstrap_two_mean), 0,
                               'bootstrap two error')

    def test_calc_diff(self):
        bootstrap_diff_mean = np.mean(np.loadtxt('bootstrap_diff.txt'))
        self.hypo.calc_bootstrap()
        self.hypo.calc_diff()
        self.assertAlmostEqual(float(np.mean(self.hypo.bootstrap_diff.means)), float(bootstrap_diff_mean), 0,
                               'bootstrap diff error')

    def test_calc_p(self):
        np.random.seed(42)
        self.hypo.calc_bootstrap()
        self.hypo.calc_diff()
        self.hypo.calc_null()
        p = self.hypo.calc_p(null_type=condition)
        print(f'p_val = {p}')
        if condition == '==':
            self.assertAlmostEqual(p, 0.547)
        elif condition == '>=':
            self.assertAlmostEqual(p, 0.742)
        elif condition == '<=':
            self.assertAlmostEqual(p, 0.258)
        else:
            self.assert_(True, "calc_p method: Hypothesis null_type must be '>=', '<=' or '=='")

    def test_eval(self):
        np.random.seed(42)
        p_val = self.hypo.eval(iterations=1000, samples=100, replacement=True, null_type=condition)
        if condition == '==':
            self.assertAlmostEqual(p_val, 0.547)
        elif condition == '>=':
            self.assertAlmostEqual(p_val, 0.742)
        elif condition == '<=':
            self.assertAlmostEqual(p_val, 0.258)
        else:
            self.assert_(True, "eval method: Hypothesis null_type must be '>=', '<=' or '=='")

    def test_plot_samples(self):
        self.hypo.calc_bootstrap()
        self.hypo.calc_diff()
        self.hypo.plot_hist_samples()
        plt.show()

    def test_plot_diff(self):
        self.hypo.calc_bootstrap()
        self.hypo.calc_diff()
        self.hypo.calc_null()
        self.hypo.plot_hist_diff(null_type=condition)
        plt.show()


if __name__ == '__main__':
    unittest.main()
