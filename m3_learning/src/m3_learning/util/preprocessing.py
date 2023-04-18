"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np

class global_scaler:
    """Class that computes the global scaler of a dataset.
    This assumes that all values are considered as part of the scaling
    """
    
    def fit(self, data):
        """Conducts the global scaler fit

        Args:
            data (np.array): data to conduct scaler
        """
        self.mean = np.mean(data.reshape(-1))
        self.std = np.std(data.reshape(-1))

    def fit_transform(self, data):
        """Conducts the fit transform

        Args:
            data (np.array): data to conduct scaler

        Returns:
            np.array: scaled data output
        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        """Applies the transform

        Args:
            data (np.array): data to conduct scaler

        Returns:
            np.array: scaled data output
        """
        return (data - self.mean)/self.std

    def inverse_transform(self, data):
        """Applies the inverse transform

        Args:
            data (np.array): data to conduct inverse scaler

        Returns:
            np.array: unscaled data output
        """
        return (data * self.std) + self.mean