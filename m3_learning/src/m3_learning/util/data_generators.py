import numpy as np
import torch
import torch.nn as nn

tanh = nn.Tanh()
selu = nn.SELU()
sigmoid = nn.Sigmoid()

def default_nl_function(t, x, y, z):
  """defines a default non-linear function

  Args:
      t (np.array): time steps to calculate
      x (np.array): parameter `x`
      y (np.array): parameter `y`
      z (np.array): parameter `z`

  Returns:
      np.array: Output spectra
  """
  # returns a function from variables
  return tanh(torch.tensor(20*(t - 2*(x-.5)))) + selu(torch.tensor((t-2*(y-0.5)))) + sigmoid(torch.tensor(-20*(t-(z-0.5))))

def generate_data(values, function=default_nl_function, length=25, range_=[-1, 1]):
  """function to generate data based on an arbitrary function

  Args:
      values (array): Input values to use
      function (obj, optional): Function to use for generation. Defaults to default_nl_function.
      length (int, optional): length of the vector to generate. Defaults to 25.
      range_ (list, optional): range of spectra where you generate data. Defaults to [-1, 1].

  Returns:
      array: computed spectra
  """
  # build x vector
  x = np.linspace(range_[0], range_[1], length)

  data = np.zeros((values.shape[0], length))

  for i in range(values.shape[0]):
      data[i, :] = function(x, values[i, 0], values[i, 1], values[i, 2])

  return data