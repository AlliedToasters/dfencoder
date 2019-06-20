from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import tqdm

from dataframe import EncoderDataFrame

def ohe(input_vector, dim):
    """Does one-hot encoding of input vector."""
    batch_size = len(input_vector)
    nb_digits = dim

    y = input_vector.reshape(-1, 1)
    y_onehot = torch.FloatTensor(batch_size, nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot

def compute_embedding_size(n_categories):
    """
    Applies a standard formula to choose the number of feature embeddings
    to use in a given embedding layers.

    n_categories is the number of unique categories in a column.
    """
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)

class AutoEncoder(torch.nn.Module):

    def __init__(self,
        hidden_layers=None,
        min_cats=10,
        swap_p=.15,
        lr=0.01,
        batch_size=256,
        optimizer='adam',
        activation='relu',
        verbose=True,
        *args,
        **kwargs
            ):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.numeric_fts = OrderedDict()
        self.binary_fts = OrderedDict()
        self.categorical_fts = OrderedDict()
        self.hidden_layers = hidden_layers
        self.min_cats = min_cats
        self.layers = []

        self.swap_p = swap_p
        self.batch_size = batch_size

        self.numeric_output = None
        self.binary_output = None

        self.num_names = []
        self.bin_names = []

        self.activation = self.interpret_activation(activation)
        self.optimizer_name = optimizer
        self.lr = lr
        self.optim = None

        self.mse = torch.nn.modules.loss.MSELoss()
        self.bce = torch.nn.modules.loss.BCELoss()
        self.cce = torch.nn.modules.loss.CrossEntropyLoss()

        self.verbose = verbose

    def interpret_activation(self, activation):
        if activation=='relu':
            return torch.relu
        elif activation=='sigmoid':
            return torch.sigmoid
        elif activation=='tanh':
            return torch.tanh
        else:
            msg = f'activation {activation} not understood. \n'
            msg += 'please use one of: \n'
            msg += "'relu', 'sigmoid', 'tanh'"

    def init_numeric(self, df):
        dt = df.dtypes
        numeric = []
        numeric += list(dt[dt==int].index)
        numeric += list(dt[dt==float].index)

        for ft in numeric:
            feature = {
                'mean':df[ft].mean(),
                'std':df[ft].std()
            }
            self.numeric_fts[ft] = feature

        self.num_names = list(self.numeric_fts.keys())

    def init_cats(self, df):
        dt = df.dtypes
        objects = list(dt[dt==pd.Categorical].index)
        for ft in objects:
            feature = {}
            vl = df[ft].value_counts()
            if len(vl) < 3:
                feature['cats'] = list(vl.index)
                self.binary_fts[ft] = feature
                continue
            cats = list(vl[vl >= self.min_cats].index)
            feature['cats'] = cats
            self.categorical_fts[ft] = feature

    def init_binary(self, df):
        dt = df.dtypes
        binaries = list(dt[dt==bool].index)
        for ft in self.binary_fts:
            feature = self.binary_fts[ft]
            for i, cat in enumerate(feature['cats']):
                feature[cat] = bool(i)
        for ft in binaries:
            feature = dict()
            feature['cats'] = [True, False]
            feature[True] = True
            feature[False] = False
            self.binary_fts[ft] = feature

        self.bin_names = list(self.binary_fts.keys())

    def init_features(self, df):
        self.init_numeric(df)
        self.init_cats(df)
        self.init_binary(df)

    def build_inputs(self):
        #will compute total number of inputs
        input_dim = 0

        #create categorical variable embedding layers
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            n_cats = len(feature['cats']) + 1
            embed_dim = compute_embedding_size(n_cats)
            embed_layer = torch.nn.Embedding(n_cats, embed_dim)
            feature['embedding'] = embed_layer
            self.add_module(f'{ft} embedding', embed_layer)
            #track embedding inputs
            input_dim += embed_dim

        #include numeric and binary fts
        input_dim += len(self.numeric_fts)
        input_dim += len(self.binary_fts)

        return input_dim

    def build_outputs(self, dim):
        self.numeric_output = torch.nn.Linear(dim, len(self.numeric_fts))
        self.binary_output = torch.nn.Linear(dim, len(self.binary_fts))

        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            cats = feature['cats']
            layer = torch.nn.Linear(dim, len(cats)+1)
            feature['output_layer'] = layer
            self.add_module(f'{ft} output', layer)

    def prepare_df(self, df):
        """
        Does data preparation on copy of input dataframe.
        Returns copy.
        """
        output_df = EncoderDataFrame()
        for ft in self.numeric_fts:
            feature = self.numeric_fts[ft]
            col = df[ft].fillna(feature['mean'])
            col -= feature['mean']
            col /= feature['std']
            output_df[ft] = col

        for ft in self.binary_fts:
            feature = self.binary_fts[ft]
            output_df[ft] = df[ft].apply(lambda x: feature.get(x, False))
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            col = pd.Categorical(df[ft], categories=feature['cats']+['_other'])
            col = col.fillna('_other')
            output_df[ft] = col
        return output_df

    def build_model(self, df):
        """
        Takes a pandas dataframe as input.
        Builds autoencoder model.

        Returns the dataframe after making changes.
        """
        #get metadata from features
        self.init_features(df)
        input_dim = self.build_inputs()

        #do the actual model building
        if self.hidden_layers is None:
            self.hidden_layers = [int(np.sqrt(2)*input_dim)]

        for i, dim in enumerate(self.hidden_layers):
            layer = torch.nn.Linear(input_dim, dim)
            input_dim = dim
            self.layers.append(layer)

        #set up predictive outputs
        self.build_outputs(dim)

        #get optimizer
        self.optim = torch.optim.Adam(self.parameters())

        #returns a copy of preprocessed dataframe.
        return self.prepare_df(df)

    def compute_targets(self, df):
        num = torch.tensor(df[self.num_names].values).float()
        bin = torch.tensor(df[self.bin_names].astype(int).values).float()
        codes = []
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            code = torch.tensor(df[ft].cat.codes.astype(int).values)
            codes.append(code)
        return num, bin, codes

    def encode_input(self, df):
        """
        Handles raw df inputs.
        Passes categories through embedding layers.
        """
        num, bin, codes = self.compute_targets(df)
        embeddings = []
        for i, ft in enumerate(self.categorical_fts):
            feature = self.categorical_fts[ft]
            emb = feature['embedding'](codes[i])
            embeddings.append(emb)
        return [num], [bin], embeddings

    def compute_outputs(self, x):
        num = self.numeric_output(x)
        bin = self.binary_output(x)
        bin = torch.sigmoid(bin)
        cat = []
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            out = feature['output_layer'](x)
            cat.append(out)
        return num, bin, cat

    def forward(self, df):
        """We do the thang. Takes pandas dataframe as input."""
        num, bin, embeddings = self.encode_input(df)
        x = torch.cat(num + bin + embeddings, dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        num, bin, cat = self.compute_outputs(x)
        return num, bin, cat

    def compute_loss(self, num, bin, cat, target_df):
        net_loss = 0
        num_target, bin_target, codes = self.compute_targets(target_df)
        mse_loss = self.mse(num, num_target)
        net_loss += mse_loss.item()
        bce_loss = self.bce(bin, bin_target)
        net_loss += bce_loss.item()
        cce_loss = []
        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], codes[i])
            cce_loss.append(loss)
            net_loss += loss.item()
        return mse_loss, bce_loss, cce_loss, net_loss

    def do_backward(self, mse, bce, cce):

        mse.backward(retain_graph=True)
        bce.backward(retain_graph=True)
        for i, ls in enumerate(cce):
            if i == len(cce)-1:
                ls.backward(retain_graph=False)
            else:
                ls.backward(retain_graph=True)

    def compute_baseline_performance(self, in_, out_):
        """
        Baseline performance is computed by generating a strong
            prediction for the identity function (predicting input==output)
            with a swapped (noisy) input,
            and computing the loss against the unaltered original data.

        This should be roughly the loss we expect when the encoder degenerates
            into the identity function solution.

        Returns net loss on baseline performance computation
            (sum of all losses)
        """
        num_pred, bin_pred, codes = self.compute_targets(in_)
        bin_pred += ((bin_pred == 0).float() * 0.05)
        bin_pred -= ((bin_pred == 1).float() * 0.05)
        codes_pred = []
        for i, cd in enumerate(codes):
            feature = list(self.categorical_fts.items())[i][1]
            dim = len(feature['cats']) + 1
            pred = ohe(cd, dim) * 5
            codes_pred.append(pred)
        mse_loss, bce_loss, cce_loss, net_loss = self.compute_loss(num_pred, bin_pred, codes_pred, out_)
        return net_loss

    def train(self, df, epochs=1, val=None):
        """Does training."""
        if self.optim is None:
            df = self.build_model(df)
        else:
            df = self.prepare_df(df)

        if val is not None:
            val_df = self.prepare_df(val)
            val_in = val_df.swap(likelihood=self.swap_p)
            msg = "Validating during training.\n"
            msg += "Computing baseline performance..."
            baseline = self.compute_baseline_performance(val_in, val_df)

        n_updates = len(df)//self.batch_size
        if len(df) % self.batch_size > 0:
            n_updates += 1
        for i in range(epochs):
            if self.verbose:
                print(f'training epoch {i+1}...')
            df = df.sample(frac=1.0)
            df = EncoderDataFrame(df)
            input_df = df.swap(likelihood=self.swap_p)
            for j in tqdm.trange(n_updates, disable=not self.verbose):
                start = j * self.batch_size
                stop = (j+1) * self.batch_size
                in_sample = input_df.iloc[start:stop]
                target_sample = df.iloc[start:stop]
                num, bin, cat = self.forward(in_sample)
                mse, bce, cce, net_loss = self.compute_loss(num, bin, cat, target_sample)
                self.do_backward(mse, bce, cce)
                self.optim.step()
                self.optim.zero_grad()

            if val is not None:
                with torch.no_grad():
                    msg = ''
                    num, bin, cat = self.forward(val_in)
                    _, _, _, net_loss = self.compute_loss(num, bin, cat, val_df)
                    msg += 'net validation loss, swapped input: \n'
                    msg += f"{round(net_loss, 4)} \n\n"

                    msg += 'baseline validation loss: '
                    msg += f"{round(baseline, 4)} \n\n"

                    num, bin, cat = self.forward(val_df)
                    _, _, _, net_loss = self.compute_loss(num, bin, cat, val_df)
                    msg += 'net validation loss, unaltered input: \n'
                    msg += f"{round(net_loss, 4)} \n\n\n"

                    if self.verbose:
                        print(msg)

    def get_vector(self, df, layer=-1):
        """
        Computes latent feature vector from hidden layer
        "layer" (argument layer=-1 returns from last hidden layer)
        """
        if self.optim is None:
            df = self.build_model(df)
        else:
            df = self.prepare_df(df)
        num, bin, embeddings = self.encode_input(df)
        x = torch.cat(num + bin + embeddings, dim=1)
        if layer == -1:
            layer = len(self.layers)
        for i in range(layer):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x
