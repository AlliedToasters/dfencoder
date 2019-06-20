import unittest
import time
import pandas as pd
import numpy as np
from dataframe import EncoderDataFrame
from autoencoder import AutoEncoder, compute_embedding_size

class TimedCase(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

class AutoEncoderTest(TimedCase):

    def test_compute_embedding_size(self):
        result = compute_embedding_size(5)
        assert result == 4

    def test_init(self):
        encoder = AutoEncoder()
        return encoder

    def test_init_numeric(self):
        encoder = AutoEncoder()
        encoder.init_numeric(df)
        assert len(encoder.numeric_fts) == 6
        assert len(encoder.numeric_fts) == len(encoder.num_names)

    def test_init_cats(self):
        encoder = AutoEncoder()
        encoder.init_cats(df)
        assert len(encoder.categorical_fts) == 7
        return encoder

    def test_init_binaries(self):
        df['mybin'] = np.random.randint(2, size=len(df)).astype(bool)
        encoder = self.test_init_cats()
        encoder.init_binary(df)
        assert len(encoder.binary_fts) == 3
        assert len(encoder.binary_fts) == len(encoder.bin_names)
        del df['mybin']

    def test_init_features(self):
        encoder = AutoEncoder()
        encoder.init_features(df)
        assert len(encoder.binary_fts) == 2
        assert len(encoder.categorical_fts) == 7
        assert len(encoder.numeric_fts) == 6

    def test_build_inputs(self):
        encoder = AutoEncoder()
        encoder.categorical_fts = {
            'ft1' : {
                'cats':['test1', 'test2', 'test3', 'test4']
            }
        }
        encoder.build_inputs()

    def test_build_model(self):
        encoder = AutoEncoder(hidden_layers=[128, 128])
        out_df = encoder.build_model(df)
        assert not out_df.isna().any().any()
        return encoder, out_df

    def test_encode_input(self):
        encoder, out_df = self.test_build_model()
        sample = out_df.sample(32)
        out = encoder.encode_input(sample)
        return encoder, sample

    def test_forward(self):
        encoder, sample = self.test_encode_input()
        num, bin, cat = encoder.forward(sample)
        assert num.shape == (32, 6)
        assert bin.shape == (32, 2)
        assert len(cat) == 7
        return encoder, num, bin, cat, sample

    def test_compute_loss(self):
        encoder, num, bin, cat, sample = self.test_forward()
        mse_loss, bce_loss, cce_loss, net_loss = encoder.compute_loss(num, bin, cat, sample)

    def test_train(self):
        encoder = AutoEncoder(verbose=False)
        sample = df.sample(511)
        encoder.train(sample, epochs=2)
        return encoder

    def test_get_vector(self):
        encoder = AutoEncoder()
        sample = df.sample(1024)
        z = encoder.get_vector(sample)
        assert z.shape[0] == 1024
        assert z.shape[1] > 1

    def test_compute_baseline_performance(self):
        encoder = AutoEncoder()
        encoder.init_features(df)
        sample = df.sample(1000)
        in_ = EncoderDataFrame(sample).swap()
        out_ = sample
        in_ = encoder.prepare_df(in_)
        out_ = encoder.prepare_df(out_)
        baseline = encoder.compute_baseline_performance(in_, out_)

class EncoderDataFrameTest(TimedCase):

    def test_init(self):
        ef = EncoderDataFrame()
        ef['test1'] = [0,2,3]
        ef['test2'] = ['a','b', 'c']

    def test_swap(self):
        ef = EncoderDataFrame(df)
        scr = ef.swap()
        assert (scr == ef).any().all()
        assert (scr != ef).any().all()
        assert not (scr == ef).all().all()

if __name__ == '__main__':
    df = pd.read_csv('adult.csv')
    unittest.main()
