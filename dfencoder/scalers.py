import numpy as np
from sklearn.preprocessing import QuantileTransformer

class StandardScaler(object):
    """Impliments standard (mean/std) scaling."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean()
        self.std = x.std()

    def transform(self, x):
        result = x.astype(float)
        result -= self.mean
        result /= self.std
        return result

    def inverse_transform(self, x):
        result = x.astype(float)
        result *= self.std
        result += self.mean
        return result

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class GaussRankScaler(object):
    """
    So-called "Gauss Rank" scaling.
    Forces a transformation, uses bins to perform
        inverse mapping.

    Uses sklearn QuantileTransformer to work.
    """

    def __init__(self):
        self.transformer = QuantileTransformer(output_distribution='normal')

    def fit(self, x):
        x = x.reshape(-1, 1)
        self.transformer.fit(x)

    def transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.transform(x)
        return result.reshape(-1)

    def inverse_transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.inverse_transform(x)
        return result.reshape(-1)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class NullScaler(object):

    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def fit_transform(self, x):
        return self.transform(x)
