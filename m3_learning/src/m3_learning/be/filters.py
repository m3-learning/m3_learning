import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline


def clean_interpolate(arr, axis=0):
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


# def clean_interpolate(data, fit_type='spline'):
#     """
#     Function which removes bad data points
#     Parameters
#     ----------
#     data : numpy, float
#         data to clean
#     fit_type : string  (optional)
#         sets the type of fitting to use
#     Returns
#     -------
#     data : numpy, float
#         cleaned data
#     """

#     shape = data.shape

#     for i in range(data.shape[2]):

#         local_shape = data[:, :, i].shape

#         # sets all non finite values to nan
#         data[~np.isfinite(data)] = np.nan
#         # function to interpolate missing points
#         data = interpolate_missing_points(data, fit_type)

#     return data


# def interpolate_missing_points(data, fit_type='spline'):
#     """
#     Interpolates bad pixels in piezoelectric hysteresis loops.\n
#     The interpolation of missing points allows for machine learning operations
#     Parameters
#     ----------
#     data : numpy array
#         array of loops
#     fit_type : string (optional)
#         selection of type of function for interpolation
#     Returns
#     -------
#     data_cleaned : numpy array
#         array of loops
#     """

#     # reshapes the data such that it can run with different data sizes
#     if data.ndim == 2:
#         data = data.reshape(np.sqrt(data.shape[0]).astype(int),
#                             np.sqrt(data.shape[0]).astype(int), -1)
#         data = np.expand_dims(data, axis=3)
#     elif data.ndim == 3:
#         data = np.expand_dims(data, axis=3)

#     # creates a vector of the size of the data
#     point_values = np.linspace(0, 1, data.shape[2])

#     # Loops around the x index
#     for i in range(data.shape[0]):

#         # Loops around the y index
#         for j in range(data.shape[1]):

#             # Loops around the number of cycles
#             for k in range(data.shape[3]):

#                 if any(~np.isfinite(data[i, j, :, k])):

#                     # selects the index where values are nan
#                     ind = np.where(np.isnan(data[i, j, :, k]))

#                     # if the first value is 0 copies the second value
#                     if 0 in np.asarray(ind):
#                         data[i, j, 0, k] = data[i, j, 1, k]

#                     # selects the values that are not nan
#                     true_ind = np.where(~np.isnan(data[i, j, :, k]))

#                     # for a spline fit
#                     if fit_type == 'spline':
#                         # does spline interpolation
#                         spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
#                                                                           data[i, j, true_ind, k].squeeze())
#                         data[i, j, ind, k] = spline(point_values[ind])

#                     # for a linear fit
#                     elif fit_type == 'linear':

#                         # does linear interpolation
#                         data[i, j, :, k] = np.interp(point_values,
#                                                      point_values[true_ind],
#                                                      data[i, j, true_ind, k].squeeze())

#     return data.squeeze()
