import numpy as np


class STEM_Dataset:
    """Class for the STEM dataset.
    """

    def __init__(self, data_path):
        """Initialization of the class.

        Args:
            data_path (string): path where the hyperspy file is located
        """
        # lazy import so the package imports without hyperspy installed
        import hyperspy.api as hs

        # loads the data
        s = hs.load(data_path,
                    reader="hspy",
                    lazy=False,
                    )

        # extracts the data
        self.data = s.data

        # sets the log data
        self.log_data = s

    @property
    def log_data(self):
        return self._log_data

    @log_data.setter
    def log_data(self, log_data):
        # add 1 to avoid log(0)
        self._log_data = np.log(log_data.data + 1)
