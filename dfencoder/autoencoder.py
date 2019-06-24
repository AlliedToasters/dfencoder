from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import tqdm

from .dataframe import EncoderDataFrame
from .logging import BasicLogger, IpynbLogger
from .scalers import StandardScaler, NullScaler, GaussRankScaler

def ohe(input_vector, dim, device="cpu"):
    """Does one-hot encoding of input vector."""
    batch_size = len(input_vector)
    nb_digits = dim

    y = input_vector.reshape(-1, 1)
    y_onehot = torch.FloatTensor(batch_size, nb_digits).to(device)

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

class CompleteLayer(torch.nn.Module):
    """
    Impliments a layer with linear transformation
    and optional activation and dropout."""

    def __init__(
            self,
            in_dim,
            out_dim,
            activation=None,
            dropout=None,
            *args,
            **kwargs
        ):
        super(CompleteLayer, self).__init__(*args, **kwargs)
        self.layers = []
        linear = torch.nn.Linear(in_dim, out_dim)
        self.layers.append(linear)
        self.add_module('linear_layer', linear)
        if activation is not None:
            self.layers.append(activation)
        if dropout is not None:
            dropout_layer = torch.nn.Dropout(dropout)
            self.layers.append(dropout_layer)
            self.add_module('dropout', dropout_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AutoEncoder(torch.nn.Module):

    def __init__(
            self,
            encoder_layers=None,
            decoder_layers=None,
            encoder_dropout=None,
            decoder_dropout=None,
            activation='relu',
            min_cats=10,
            swap_p=.15,
            lr=0.01,
            batch_size=256,
            eval_batch_size=1024,
            optimizer='adam',
            amsgrad=False,
            momentum=0,
            betas=(0.9, 0.999),
            dampening=0,
            weight_decay=0,
            lr_decay=None,
            nesterov=False,
            verbose=False,
            device=None,
            logger='basic',
            progress_bar=True,
            scaler='standard',
            *args,
            **kwargs
        ):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.numeric_fts = OrderedDict()
        self.binary_fts = OrderedDict()
        self.categorical_fts = OrderedDict()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.min_cats = min_cats
        self.encoder = []
        self.decoder = []
        self.train_mode = self.train

        self.swap_p = swap_p
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.numeric_output = None
        self.binary_output = None

        self.num_names = []
        self.bin_names = []

        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.amsgrad=amsgrad
        self.momentum=momentum
        self.betas=betas
        self.dampening=dampening
        self.weight_decay=weight_decay
        self.nesterov=nesterov
        self.optim = None
        self.progress_bar = progress_bar

        self.mse = torch.nn.modules.loss.MSELoss(reduction='none')
        self.bce = torch.nn.modules.loss.BCELoss(reduction='none')
        self.cce = torch.nn.modules.loss.CrossEntropyLoss(reduction='none')

        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.logger = logger
        if scaler == 'standard':
            self.Scaler = StandardScaler
        elif scaler == 'gauss_rank':
            self.Scaler = GaussRankScaler
        elif scaler is None:
            self.Scaler = NullScaler

    def interpret_activation(self):
        activation = self.activation
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
                'std':df[ft].std(),
                'scaler':self.Scaler()
            }
            feature['scaler'].fit(df[ft][~df[ft].isna()].values)
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
            trans_col = feature['scaler'].transform(col.values)
            trans_col = pd.Series(index=df.index, data=trans_col)
            output_df[ft] = trans_col

        for ft in self.binary_fts:
            feature = self.binary_fts[ft]
            output_df[ft] = df[ft].apply(lambda x: feature.get(x, False))
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            col = pd.Categorical(df[ft], categories=feature['cats']+['_other'])
            col = col.fillna('_other')
            output_df[ft] = col
        return output_df

    def build_optimizer(self):

        lr = self.lr
        params = self.parameters()
        if self.optimizer == 'adam':
            return torch.optim.Adam(
                params,
                lr=self.lr,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
                betas=self.betas
            )
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(
                params,
                lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                dampening=self.dampening,
                weight_decay=self.weight_decay,

            )

    def build_model(self, df):
        """
        Takes a pandas dataframe as input.
        Builds autoencoder model.

        Returns the dataframe after making changes.
        """
        if self.verbose:
            print('Building model...')

        #get metadata from features
        self.init_features(df)
        input_dim = self.build_inputs()

        #do the actual model building
        if self.encoder_layers is None:
            self.encoder_layers = [int(np.sqrt(2)*input_dim) for _ in range(2)]

        if self.decoder_layers is None:
            self.decoder_layers = []

        if self.encoder_dropout is None or type(self.encoder_dropout) == float:
            drp = self.encoder_dropout
            self.encoder_dropout = [drp for _ in self.encoder_layers]

        if self.decoder_dropout is None or type(self.decoder_dropout) == float:
            drp = self.decoder_dropout
            self.decoder_dropout = [drp for _ in self.decoder_layers]

        for i, dim in enumerate(self.encoder_layers):
            if i != len(self.encoder_layers) - 1:
                activation = self.interpret_activation()
            else:
                activation = None
            layer = CompleteLayer(
                input_dim,
                dim,
                activation = activation,
                dropout = self.encoder_dropout[i]
            )
            input_dim = dim
            self.encoder.append(layer)
            self.add_module(f'encoder_{i}', layer)

        for i, dim in enumerate(self.decoder_layers):
            if i != len(self.decoder_layers) - 1:
                activation = self.interpret_activation()
            else:
                activation = None
            layer = CompleteLayer(
                input_dim,
                dim,
                activation = activation,
                dropout = self.decoder_dropout[i]
            )
            input_dim = dim
            self.decoder.append(layer)
            self.add_module(f'decoder_{i}', layer)

        #set up predictive outputs
        self.build_outputs(dim)

        #get optimizer
        self.optim = self.build_optimizer()
        if self.lr_decay is not None:
            self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optim, self.lr_decay)

        cat_names = list(self.categorical_fts.keys())
        fts = self.num_names + self.bin_names + cat_names
        if self.logger == 'basic':
            self.logger = BasicLogger(fts=fts)
        elif self.logger == 'ipynb':
            self.logger = IpynbLogger(fts=fts)
        #returns a copy of preprocessed dataframe.
        self.to(self.device)

        if self.verbose:
            print('done!')
        return self.prepare_df(df)

    def compute_targets(self, df):
        num = torch.tensor(df[self.num_names].values).float().to(self.device)
        bin = torch.tensor(df[self.bin_names].astype(int).values).float().to(self.device)
        codes = []
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            code = torch.tensor(df[ft].cat.codes.astype(int).values).to(self.device)
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

    def encode(self, x, layers=None):
        if layers is None:
            layers = len(self.encoder)
        for i in range(layers):
            layer = self.encoder[i]
            x = layer(x)
        return x

    def decode(self, x, layers=None):
        if layers is None:
            layers = len(self.decoder)
        for i in range(layers):
            layer = self.decoder[i]
            x = layer(x)
        num, bin, cat = self.compute_outputs(x)
        return num, bin, cat

    def forward(self, df):
        """We do the thang. Takes pandas dataframe as input."""
        num, bin, embeddings = self.encode_input(df)
        x = torch.cat(num + bin + embeddings, dim=1)

        encoding = self.encode(x)
        num, bin, cat = self.decode(encoding)

        return num, bin, cat

    def compute_loss(self, num, bin, cat, target_df, logging=True, _id=False):
        if logging:
            if self.logger is not None:
                logging = True
            else:
                logging = False
        net_loss = []
        num_target, bin_target, codes = self.compute_targets(target_df)
        mse_loss = self.mse(num, num_target)
        net_loss += list(mse_loss.mean(dim=0).cpu().detach().numpy())
        mse_loss = mse_loss.mean()
        bce_loss = self.bce(bin, bin_target)
        net_loss += list(bce_loss.mean(dim=0).cpu().detach().numpy())
        bce_loss = bce_loss.mean()
        cce_loss = []
        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], codes[i])
            loss = loss.mean()
            cce_loss.append(loss)
            val = loss.cpu().item()
            net_loss += [val]
        if logging:
            if self.training:
                self.logger.training_step(net_loss)
            elif _id:
                self.logger.id_val_step(net_loss)
            elif not self.training:
                self.logger.val_step(net_loss)

        net_loss = np.array(net_loss).mean()
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
        self.eval()
        num_pred, bin_pred, codes = self.compute_targets(in_)
        bin_pred += ((bin_pred == 0).float() * 0.05)
        bin_pred -= ((bin_pred == 1).float() * 0.05)
        codes_pred = []
        for i, cd in enumerate(codes):
            feature = list(self.categorical_fts.items())[i][1]
            dim = len(feature['cats']) + 1
            pred = ohe(cd, dim, device=self.device) * 5
            codes_pred.append(pred)
        mse_loss, bce_loss, cce_loss, net_loss = self.compute_loss(
            num_pred,
            bin_pred,
            codes_pred,
            out_,
            logging=False
        )
        if isinstance(self.logger, BasicLogger):
            self.logger.baseline_loss = net_loss
        return net_loss

    def fit(self, df, epochs=1, val=None):
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
            if self.verbose:
                print(msg)
            result = []
            val_batches = len(val_df)//self.eval_batch_size
            if len(val_df) % self.eval_batch_size != 0:
                val_batches += 1

        n_updates = len(df)//self.batch_size
        if len(df) % self.batch_size > 0:
            n_updates += 1
        for i in range(epochs):
            self.train()
            if self.verbose:
                print(f'training epoch {i+1}...')
            df = df.sample(frac=1.0)
            df = EncoderDataFrame(df)
            input_df = df.swap(likelihood=self.swap_p)
            for j in tqdm.trange(n_updates, disable=not self.progress_bar):
                start = j * self.batch_size
                stop = (j+1) * self.batch_size
                in_sample = input_df.iloc[start:stop]
                target_sample = df.iloc[start:stop]
                num, bin, cat = self.forward(in_sample)
                mse, bce, cce, net_loss = self.compute_loss(
                    num, bin, cat, target_sample,
                    logging=True
                )
                self.do_backward(mse, bce, cce)
                self.optim.step()
                self.optim.zero_grad()

            if self.lr_decay is not None:
                self.lr_decay.step()

            if val is not None:
                self.eval()
                with torch.no_grad():
                    swapped_loss = []
                    id_loss = []
                    for i in range(val_batches):
                        start = i * self.eval_batch_size
                        stop = (i+1) * self.eval_batch_size

                        slc_in = val_in.iloc[start:stop]
                        slc_out = val_df.iloc[start:stop]

                        num, bin, cat = self.forward(slc_in)
                        _, _, _, net_loss = self.compute_loss(num, bin, cat, slc_out)
                        swapped_loss.append(net_loss)


                        num, bin, cat = self.forward(slc_out)
                        _, _, _, net_loss = self.compute_loss(num, bin, cat, slc_out, _id=True)
                        id_loss.append(net_loss)

                    self.logger.end_epoch()
                    if self.verbose:
                        swapped_loss = np.array(swapped_loss).mean()
                        id_loss = np.array(id_loss).mean()

                        msg = '\n'
                        msg += 'net validation loss, swapped input: \n'
                        msg += f"{round(swapped_loss, 4)} \n\n"
                        msg += 'baseline validation loss: '
                        msg += f"{round(baseline, 4)} \n\n"
                        msg += 'net validation loss, unaltered input: \n'
                        msg += f"{round(id_loss, 4)} \n\n\n"
                        print(msg)

    def get_representation(self, df, layer=0):
        """
        Computes latent feature vector from hidden layer
            given input dataframe.

        argument layer (int) specifies which layer to get.
        by default (layer=0), returns the "encoding" layer.
            layer < 0 counts layers back from encoding layer.
            layer > 0 counts layers forward from encoding layer.
        """
        result = []
        n_batches = len(df)//self.eval_batch_size
        if len(df) % self.eval_batch_size != 0:
            n_batches += 1

        self.eval()
        if self.optim is None:
            df = self.build_model(df)
        else:
            df = self.prepare_df(df)
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.eval_batch_size
                stop = (i+1) * self.eval_batch_size
                num, bin, embeddings = self.encode_input(df.iloc[start:stop])
                x = torch.cat(num + bin + embeddings, dim=1)
                if layer <= 0:
                    layers = len(self.encoder) + layer
                    x = self.encode(x, layers=layers)
                else:
                    x = self.encode(x)
                    x = self.decode(x, layers=layer)
                result.append(x.cpu().numpy())
        z = np.concatenate(result, axis=0)
        return z

    def get_anomaly_score(self, df):
        """
        Returns a per-row loss of the input dataframe.
        Does not corrupt inputs.
        """
        self.eval()
        data = self.prepare_df(df)
        num_target, bin_target, codes = self.compute_targets(data)
        with torch.no_grad():
            num, bin, cat = self.forward(data)


        mse_loss = self.mse(num, num_target)
        net_loss = mse_loss.cpu().data.sum(dim=1)
        bce_loss = self.bce(bin, bin_target)
        net_loss += bce_loss.cpu().data.sum(dim=1)
        cce_loss = []
        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], codes[i])
            cce_loss.append(loss)
            net_loss += loss.cpu().data

        return net_loss

    def decode_to_df(self, x, df=None):
        """
        Runs input embeddings through decoder
        and converts outputs into a dataframe.
        """
        if df is None:
            cols = [x for x in self.binary_fts.keys()]
            cols += [x for x in self.numeric_fts.keys()]
            cols += [x for x in self.categorical_fts.keys()]
            df = pd.DataFrame(index=range(len(x)), columns=cols)

        num, bin, cat = self.decode(x)

        num_cols = [x for x in self.numeric_fts.keys()]
        num_df = pd.DataFrame(data=num.cpu().numpy(), index=df.index)
        num_df.columns = num_cols
        for ft in num_df.columns:
            feature = self.numeric_fts[ft]
            col = num_df[ft]
            trans_col = feature['scaler'].inverse_transform(col.values)
            result = pd.Series(index=df.index, data=trans_col)
            num_df[ft] = result

        bin_cols = [x for x in self.binary_fts.keys()]
        bin_df = pd.DataFrame(data=bin.cpu().numpy(), index=df.index)
        bin_df.columns = bin_cols
        bin_df = bin_df.apply(lambda x: round(x)).astype(bool)
        for ft in bin_df.columns:
            feature = self.binary_fts[ft]
            map = {
                False:feature['cats'][0],
                True:feature['cats'][1]
            }
            bin_df[ft] = bin_df[ft].apply(lambda x: map[x])

        cat_df = pd.DataFrame(index=df.index)
        for i, ft in enumerate(self.categorical_fts):
            feature = self.categorical_fts[ft]
            #get argmax excluding NaN column (impute with next-best guess)
            codes = torch.argmax(cat[i][:, :-1], dim=1).cpu().numpy()
            cat_df[ft] = codes
            cats = feature['cats']
            cat_df[ft] = cat_df[ft].apply(lambda x: cats[x])

        #concat
        output_df = pd.concat([num_df, bin_df, cat_df], axis=1)

        return output_df[df.columns]

    def df_predict(self, df):
        """
        Runs end-to-end model.
        Interprets output and creates a dataframe.
        Outputs dataframe with same shape as input
            containing model predictions.
        """
        self.eval()
        data = self.prepare_df(df)
        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            x = self.encode(x)
            output_df = self.decode_to_df(x, df=df)

        return output_df
