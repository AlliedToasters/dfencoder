import pandas as pd
import numpy as np

class EncoderDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(EncoderDataFrame, self).__init__(*args, **kwargs)

    def swap(self, likelihood=.15):
        """
        Performs random swapping of data.
        Each value has a likelihood of *argument likelihood*
            of being randomly replaced with a value from a different
            row.
        Returns a copy of the dataframe with equal size.
        """
        #copy initial df for preservation of true values
        result = self.copy()

        #select values to swap
        donors = np.random.random(size=(self.__len__(), len(self.columns)))
        donors = np.where(donors < likelihood, True, False)
        changers = donors.copy()
        np.random.shuffle(changers)

        #and swap them
        for i, col in enumerate(self.columns):
            changer_idx = changers[:, i]
            donor_idx = donors[:, i]
            result.loc[changer_idx, col] = self[col][donor_idx].values
        return result
