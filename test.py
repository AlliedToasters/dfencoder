import unittest
import time
import os
import shutil
import json
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch

from dfencoder import EncoderDataFrame
from dfencoder import AutoEncoder, compute_embedding_size, CompleteLayer, NullIndicator
from dfencoder import BasicLogger, IpynbLogger, TensorboardXLogger
from dfencoder import StandardScaler, NullScaler, GaussRankScaler

class TimedCase(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

class ModelBuilder(object):

    def __init__(self):
        self.model = None
        self.out_df = None

    def build_model(self):
        if self.model is None:
            encoder = AutoEncoder(
                encoder_layers=[32, 32],
                decoder_layers=[32, 32],
                encoder_dropout=.5,
                decoder_dropout=[.2, None],
                activation='tanh',
                swap_p=.2,
                batch_size=123,
                optimizer='sgd',
                lr_decay=.95
            )
            encoder.build_model(df)
            out_df = encoder.prepare_df(df)
            assert not out_df.isna().any().any()
            layers_count = 0
            for prm in encoder.parameters():
                layers_count += 1
            assert layers_count == 33
            self.model, self.out_df = encoder, out_df
            return encoder, out_df
        else:
            return self.model, self.out_df

class TestCompleteLayer(TimedCase):
    def test_init(self):
        layer = CompleteLayer(12, 5, activation='sigmoid', dropout=.2)
        assert len(layer.layers) == 3
        return layer

    def test_forward(self):
        layer = self.test_init()
        x = torch.randn((34, 12))
        out = layer(x)
        assert out.shape == (34, 5)
        assert (out == 0).any().any()

    def test_interpret_activation(self):
        result = CompleteLayer.interpret_activation(None, 'leaky_relu')
        assert result == torch.nn.functional.leaky_relu

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

    def test_init_cyclical(self):
        df['mytime'] = 1539435837534561201
        df['mytime'] = pd.to_datetime(df['mytime'])
        df.loc[10, 'mytime'] = np.nan
        encoder = AutoEncoder()
        encoder.init_cyclical(df)
        assert list(encoder.cyclical_fts.keys()) == ['mytime']

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

    def test_build_vanilla_model(self):
        encoder = AutoEncoder()
        encoder.build_model(df)

    def test_build_model(self):

        encoder = AutoEncoder(
            encoder_layers=[32, 32],
            decoder_layers=[32, 32],
            encoder_dropout=.5,
            decoder_dropout=[.2, None],
            activation='tanh',
            swap_p=.2,
            batch_size=123,
            optimizer='sgd',
            lr_decay=.95
        )
        encoder.build_model(df)
        out_df = encoder.prepare_df(df)
        assert not out_df.isna().any().any()
        layers_count = 0
        for prm in encoder.parameters():
            layers_count += 1
        assert layers_count == 33
        return encoder, out_df

    def test_encode_input(self):
        encoder, out_df = self.test_build_model()
        sample = out_df.sample(32)
        out = encoder.encode_input(sample)
        return encoder, sample

    def test_forward(self):
        encoder, sample = self.test_encode_input()
        num, bin, cat = encoder.forward(sample)
        #raise Exception(num.shape)
        if 'mytime' in encoder.cyclical_fts:
            assert num.shape == (32, 15)
        else:
            assert num.shape == (32, 6)
        assert bin.shape == (32, 2)
        assert len(cat) == 7
        return encoder, num, bin, cat, sample

    def test_compute_loss(self):
        encoder, num, bin, cat, sample = self.test_forward()
        mse_loss, bce_loss, cce_loss, net_loss = encoder.compute_loss(num, bin, cat, sample)

    def test_fit(self):
        encoder = AutoEncoder(
            verbose=False,
            optimizer='sgd',
            lr=.01,
            lr_decay=.95,
            progress_bar=False,
            scaler={'age':'standard'},
        )
        df['mytime'] = 1539435837534561201
        df['mytime'] = pd.to_datetime(df['mytime'])
        sample = df.sample(511)
        encoder.fit(sample, epochs=2)
        assert isinstance(encoder.numeric_fts['age']['scaler'], StandardScaler)
        assert isinstance(encoder.numeric_fts['fnlwgt']['scaler'], GaussRankScaler)
        assert encoder.lr_decay.get_lr()[0] < .01
        anomaly_score = encoder.get_anomaly_score(sample)
        assert anomaly_score.shape == (511,)
        encoder.fit(sample, epochs=2)
        data = encoder.df_predict(sample)
        assert (data.columns == sample.columns).all()
        assert data.shape == sample.shape
        return encoder

    def test_inference(self):
        record = df.sample()
        js = record.iloc[0].to_json()
        output = model._deserialize_json(js)
        z_json = model.get_deep_stack_features_json(js)
        dct = json.loads(js)
        z_dict = model.get_deep_stack_features_json(dct)
        z = model.get_deep_stack_features(record)
        assert (z_json == z).all()
        assert (z_json == z_dict).all()

    def test_get_representation(self):
        encoder = AutoEncoder()
        sample = df.sample(1025)
        z = encoder.get_representation(sample)
        assert z.shape[0] == 1025
        assert z.shape[1] > 1
        assert isinstance(z, torch.Tensor)

    def test_get_deep_stack_features(self):
        encoder = AutoEncoder(
            encoder_layers = [50, 100, 150],
            decoder_layers = [44, 67]
        )
        sample = df.sample(1025)
        z = encoder.get_deep_stack_features(sample)
        assert z.shape[0] == 1025
        assert z.shape[1] == 411
        assert isinstance(z, torch.Tensor)

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
        cols = list(df.columns)
        if 'mytime' in cols:
            cols.remove('mytime')
        df_ = df[cols]
        ef = EncoderDataFrame(df_)
        scr = ef.swap()
        assert (scr == ef).any().all()
        assert (scr != ef).any().all()
        assert not (scr == ef).all().all()

class LoggerTest(TimedCase):

    def test_basic_logger(self):
        logger = BasicLogger(fts=['ft1', 'ft2', 'ft3'])
        self.run_logging_test(logger)

    def run_logging_test(self, logger):
        n_epochs = logger.n_epochs
        assert len(logger.train_fts) == 3
        assert len(logger.val_fts) == 3
        logger.training_step([0.2, 0.3, 0.2])
        logger.training_step([0.1, 0.1, -0.2])
        logger.val_step([0.2, 0.3, 0.2])
        logger.val_step([0.1, 0.1, -0.2])
        logger.id_val_step([0.2, 0.3, 0.2])
        logger.id_val_step([0.1, 0.1, -0.2])
        logger.end_epoch()
        assert logger.train_fts['ft3'][1][-1] == 0
        assert logger.val_fts['ft3'][1][-1] == 0
        assert logger.id_val_fts['ft3'][1][-1] == 0
        assert logger.n_epochs == n_epochs + 1

    def test_ipynb_logger(self):
        logger = IpynbLogger(fts=['ft1', 'ft2', 'ft3'], baseline_loss=0.2)
        self.run_logging_test(logger)
        logger.training_step([0.2, 0.3, 0.2])
        logger.training_step([0.1, 0.1, -0.2])
        logger.val_step([0.2, 0.3, 0.2])
        logger.val_step([0.1, 0.1, -0.2])
        logger.id_val_step([0.2, 0.3, 0.2])
        logger.id_val_step([0.1, 0.1, -0.2])
        #logger.end_epoch()

    def test_tensorboardx_logger(self):
        logger = TensorboardXLogger(logdir='_testlog/', fts=['ft1', 'ft2', 'ft3'])
        cats = OrderedDict()
        cats['cat1'] = {
            'cats': ['cow', 'horse', 'pig', 'cat'],
            'embedding': torch.nn.Embedding(5, 4),
            'output_layer': None
        }
        cats['cat2'] = {
            'cats': ['cow', 'horse', 'pig', 'cat'],
            'embedding': torch.nn.Embedding(5, 4),
            'output_layer': None
        }
        for i in range(10):
            self.run_logging_test(logger)
            logger.show_embeddings(cats)


class ScalerTest(TimedCase):

    def test_standard_scaler(self):
        scaler = StandardScaler()
        x = np.random.randn(100)
        x *= 3
        x -= 3
        x_ = scaler.fit_transform(x)
        assert np.abs(x_.mean()) < 0.01
        assert .99 < x_.std() < 1.01

    def test_null_scaler(self):
        scaler=NullScaler()
        x = np.random.randn(100)
        x *= 3
        x -= 3
        x_ = scaler.fit_transform(x)
        assert (x_ == x).all()

    def test_gauss_rank_scaler(self):
        scaler = GaussRankScaler()
        x = np.random.randn(10000)
        x *= 3
        x -= 3
        x_ = scaler.fit_transform(x)
        assert np.abs(x_.mean()) < 0.01
        assert .99 < x_.std() < 1.01

class NullIndicatorTest(TimedCase):

    def test_null_indicator(self):
        ind = NullIndicator()
        test_df = pd.DataFrame(columns=['has null', 'has no null'])
        test_df['has null'] = np.random.randn(100)
        test_df.loc[4, 'has null'] = np.nan
        test_df['has no null'] = np.random.randn(100)
        ind.fit(test_df)
        assert 'has null' in ind.fts
        assert 'has no null' not in ind.fts
        output_df = ind.transform(test_df)
        assert 'has null_was_nan' in output_df.columns
        assert output_df.loc[4, 'has null_was_nan'] == True
        ind2 = NullIndicator(required_fts=['has no null'])
        ind2.fit(test_df)
        output_df = ind2.transform(test_df)
        assert len(output_df.columns) == 4
        assert not output_df['has no null_was_nan'].any()

if __name__ == '__main__':
    os.mkdir('_testlog')
    df = pd.read_csv('adult.csv')
    b = ModelBuilder()
    model, _ = b.build_model()
    unittest.main(exit=False)
    shutil.rmtree('_testlog')
    quit()
