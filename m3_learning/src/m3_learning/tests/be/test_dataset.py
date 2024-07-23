import sys
sys.path.append('/home/ferroelectric/Documents/m3_learning/m3_learning/src')

from m3_learning.be.datasets import resample
import numpy as np
import unittest

class TestResample(unittest.TestCase):
    def test_resample(self):
        # Test case 1: Resample 1D array with 10 points to 20 points
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        num_points = 20
        expected_output = np.array([1.        , 1.52631579, 2.05263158, 2.57894737, 3.10526316,
                                   3.63157895, 4.15789474, 4.68421053, 5.21052632, 5.73684211,
                                   6.26315789, 6.78947368, 7.31578947, 7.84210526, 8.36842105,
                                   8.89473684, 9.42105263, 9.94736842, 10.47368421, 11.        ])
        resampled_y = resample(y, num_points)
        np.testing.assert_allclose(resampled_y, expected_output)

        # Test case 2: Resample 2D array with shape (5, 5) to shape (10, 10)
        y = np.array([[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15],
                      [16, 17, 18, 19, 20],
                      [21, 22, 23, 24, 25]])
        num_points = 10
        expected_output = np.array([[ 1.        ,  1.44444444,  1.88888889,  2.33333333,  2.77777778,
                                      3.22222222,  3.66666667,  4.11111111,  4.55555556,  5.        ],
                                    [ 6.        ,  6.44444444,  6.88888889,  7.33333333,  7.77777778,
                                      8.22222222,  8.66666667,  9.11111111,  9.55555556, 10.        ],
                                    [11.        , 11.44444444, 11.88888889, 12.33333333, 12.77777778,
                                     13.22222222, 13.66666667, 14.11111111, 14.55555556, 15.        ],
                                    [16.        , 16.44444444, 16.88888889, 17.33333333, 17.77777778,
                                     18.22222222, 18.66666667, 19.11111111, 19.55555556, 20.        ],
                                    [21.        , 21.44444444, 21.88888889, 22.33333333, 22.77777778,
                                     23.22222222, 23.66666667, 24.11111111, 24.55555556, 25.        ]])
        resampled_y = resample(y, num_points)
        np.testing.assert_allclose(resampled_y, expected_output)

if __name__ == '__main__':
    unittest.main()