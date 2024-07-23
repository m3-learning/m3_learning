import sys
sys.path.append('/home/ferroelectric/Documents/m3_learning/m3_learning/src')

from m3_learning.be.filters import clean_interpolate
import numpy as np
import pytest
from scipy.interpolate import CubicSpline


def test_clean_interpolate():
    # Test case 1: Interpolate along axis 0
    arr1 = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8],
                    [5, 6, 7, 8], [5, 6, 7, 8]])
    expected1 = np.array(
        [[1, 2, 7, 4], [5, 4.66666667, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]])
    assert np.allclose(clean_interpolate(arr1, axis=0), expected1)

    # Test case 2: Interpolate along axis 1
    arr2 = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8]])
    expected2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert np.allclose(clean_interpolate(arr2, axis=1), expected2)

    # Test case 3: Interpolate along axis 2 (out of bounds)
    arr3 = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8]])
    with pytest.raises(ValueError):
        clean_interpolate(arr3, axis=2)

    # Test case 4: Interpolate with no non-finite values
    arr4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert np.allclose(clean_interpolate(arr4, axis=0), expected4)

    # Test case 5: Interpolate with all finite
    arr5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert np.allclose(clean_interpolate(arr5, axis=0), expected5)


if __name__ == "__main__":
    pytest.main([__file__])
