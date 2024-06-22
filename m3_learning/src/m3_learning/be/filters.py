import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline


def clean_interpolate(arr, axis=0):
    """Function that removes bad data points by interpolating them.

    Args:
        arr (np.array): np array to clean
        axis (int, optional): axis which to interpolate along. Defaults to 0.

    Raises:
        ValueError: error that is returned if the wrong axis is selected

    Returns:
        np.array: cleaned data
    """
    # Check the axis validity
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("Axis is out of bounds for the array.")

    # Move the interpolation axis to the beginning and reshape to 2D
    new_shape = [arr.shape[axis]] + [-1]
    transposed_axes = [axis] + [i for i in range(arr.ndim) if i != axis]
    arr_reshaped = np.transpose(arr, transposed_axes).reshape(new_shape)

    # Iterate through the 2D reshaped array and interpolate along the specified axis
    for i in range(arr_reshaped.shape[1]):
        slice_ = arr_reshaped[:, i]
        finite_indices = np.where(np.isfinite(slice_))[0]
        non_finite_indices = np.where(~np.isfinite(slice_))[0]

        if len(finite_indices) < 2:
            # If there are fewer than 2 finite points, spline interpolation cannot be applied
            continue

        # Compute the cubic spline interpolation
        spline = CubicSpline(finite_indices, slice_[finite_indices])

        # Interpolate the non-finite values using the cubic spline
        interpolated_values = spline(non_finite_indices)
        slice_[non_finite_indices] = interpolated_values

    # Reshape back to the original n-dimensional shape
    interpolated_arr = arr_reshaped.reshape(
        arr.shape[axis], *arr.shape[:axis], *arr.shape[axis+1:]).transpose(np.argsort(transposed_axes))

    return interpolated_arr