import pandas as pd

class EncoderDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(EncoderDataFrame, self).__init__(*args, **kwargs)

    def scramble(likelihood=.15):
        """
        Performs random scrambling of data.
        Each value has a likelihood of arguments likelihood
            of being randomly replaced with a value from a different
            row.
        Returns a copy of the dataframe with equal size.
        """

        result = self.copy()
        return result
