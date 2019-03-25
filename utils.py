import os
import re
import gc
import pickle
import operator 
import math
import string
from collections import OrderedDict

import cv2
import numpy as np
from numpy.random import RandomState
from PIL import Image, ImageDraw
from tqdm import tqdm

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, add, \
                         Dense, Input, Lambda, Bidirectional, ZeroPadding2D, concatenate, Flatten, \
                         concatenate, multiply, ReLU, DepthwiseConv2D, TimeDistributed, MaxPool2D

from keras.layers import LSTM, GRU
# Inference only on GPU:
# from keras.layers import CuDNNLSTM as LSTM
# from keras.layers import CuDNNGRU as GRU
from keras.layers.core import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.models import Model, load_model, model_from_json
from keras import optimizers
from keras import backend as K
import tensorflow as tf

class CRNN:

    def __init__(self, num_classes=97, max_string_len=23, shape=(40,40,1), time_dense_size=128, GRU=False, n_units=256):

        self.num_classes = num_classes
        self.shape = shape
        self.max_string_len = max_string_len
        self.n_units = n_units
        self.GRU = GRU
        self.time_dense_size = time_dense_size

    def depthwise_conv_block(self, inputs, pointwise_conv_filters, conv_size=(3, 3), pooling=None):
        x = DepthwiseConv2D((3, 3), padding='same', strides=(1, 1), depth_multiplier=1, use_bias=False)(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = ReLU(6.)(x)
        x = Conv2D(pointwise_conv_filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = ReLU(6.)(x)
        if pooling is not None:
            x = MaxPooling2D(pooling)(x)
            if pooling[0] == 2:
                self.pooling_counter_h += 1
            if pooling[1] == 2:
                self.pooling_counter_w += 1
        return Dropout(0.1)(x)

    def get_model(self):
        self.pooling_counter_h, self.pooling_counter_w = 0, 0
        inputs = Input(name='the_input', shape=self.shape, dtype='float32') #100x32x1
        # spatial transformer
        x = STN(inputs, sampling_size=self.shape[:2]) #100x32x1
        x = ZeroPadding2D(padding=(2, 2))(x) #104x36x1
        x = self.depthwise_conv_block(x, 64, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 128, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 256, conv_size=(3, 3), pooling=(2, 2)) #52x18x256
        x = self.depthwise_conv_block(x, 256, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=(1, 2)) #52x9x512
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=None)

        conv_to_rnn_dims = ((self.shape[0]+4) // (2 ** self.pooling_counter_h), ((self.shape[1]+4) // (2 ** self.pooling_counter_w)) * 512)
        x = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(x) #52x4608
        x = Dense(self.time_dense_size, activation='relu', name='dense1')(x) #52x128 (time_dense_size)
        x = Dropout(0.4)(x)

        if not self.GRU:    
            x = Bidirectional(LSTM(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum', weights=None)(x)
            x = Bidirectional(LSTM(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat', weights=None)(x)
        else:
            x = Bidirectional(GRU(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum', weights=None)(x)
            x = Bidirectional(GRU(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat', weights=None)(x)
        x = Dropout(0.2)(x)

        x_ctc = Dense(self.num_classes, kernel_initializer='he_normal', name='dense2')(x)
        y_pred = Activation('softmax', name='softmax')(x_ctc)

        labels = Input(name='the_labels', shape=[self.max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        outputs = [loss_out]

        model = Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs)
        return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

########################################################################
# SPATIAL TRANSFORMER
########################################################################

from keras.engine.topology import Layer
def K_meshgrid(x, y):
    return tf.meshgrid(x, y)

def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)

class BilinearInterpolation(Layer):

    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size=(100, 32), **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        transformations = K.reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)
        return interpolated_image

    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        return config

def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights

def STN(image, sampling_size=(100, 32)):
    locnet = MaxPool2D(pool_size=(2, 2))(image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([image, locnet])
    return x

########################################################################

def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def edit_distance(y_pred, y_true):
    mean_distance, length = 0, len(y_true)
    for y0, y in zip(y_pred, y_true):
        mean_distance += levenshtein(y0, y) / length
    return mean_distance

def normalized_edit_distance(y_pred, y_true):
    mean_distance, length = 0, len(y_true)
    for y0, y in zip(y_pred, y_true):
        mean_distance += levenshtein(y0, y) / (len(y) * length)
    return mean_distance

def load_model_custom(path, weights="model"):
    json_file = open(path+'/model.json', 'r')
    loaded_model_json = json_file.read()    
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+"/%s.h5" % weights)
    return loaded_model

def init_predictor(model):
    try:
        return Model(inputs=model.input, outputs=model.get_layer('softmax').output)
    except:
        return Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('softmax').output)

def labels_to_text(labels, inverse_classes=None):
    ret = []
    for c in labels:
        if c == len(inverse_classes) or c == -1:
            ret.append("")
        else:
            ret.append(str(inverse_classes[c]))
    return "".join(ret)

def load_custom_model(model_path, model_name='/model.json', weights="/final_weights.h5"):
    json_file = open(model_path+model_name, 'r')
    loaded_model_json = json_file.read()    
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'BilinearInterpolation': BilinearInterpolation})
    model.load_weights(model_path+weights)
    return model

class DecodeCTCPred:

    def __init__(self, top_paths=1, beam_width=5, inverse_classes=None):
        self.top_paths = top_paths
        self.beam_width = beam_width
        self.inverse_classes = inverse_classes

    def labels_to_text(self, labels):
        ret = []
        for c in labels:
            if c == len(self.inverse_classes) or c == -1:
                ret.append("")
            else:
                ret.append(self.inverse_classes[c])
        return "".join(ret)

    def decode(self, result):
        results = []
        if self.beam_width < self.top_paths:
          self.beam_width = self.top_paths
        for out in result:
            out = np.expand_dims(out, axis=0)
            labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                               greedy=False, beam_width=self.beam_width, top_paths=self.top_paths)[0][0])[0]
            text = self.labels_to_text(labels)
            results.append(text)
        return results

def read_img(name):
    img = cv2.imread(name)
    img = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def open_img(img, img_size, p=.7):

    if isinstance(img, str):
        name = img
        img = read_img(name)

    img = img[::-1].T

    val, counts = np.unique(img, return_counts=True)
    fill = val[np.where(counts == counts.max())[0][0]]

    if all([img.shape[0] <= img_size[0] // 2, img.shape[1] <= img_size[1] // 2]):
        img = cv2.resize(img, (int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)), Image.LANCZOS)
        
    # randomly with probability of "p", move word inside the bbox
    if (img_size[1] - img.shape[1]) > 2 :
        delta = img_size[1]-img.shape[1]
        r = round(np.random.uniform(0,1), 1)
        if r < p and p > 0.:
            c = np.random.choice(list(range(2, delta)))
            start = np.full((img.shape[0], c - 1), fill)
            end = np.full((img.shape[0], delta - c), fill)
            img = np.concatenate([start, img, end], axis=1)
        else:
            img = np.concatenate([img, np.full((img.shape[0], delta), fill)], axis=1)
        
    if (img_size[0] - img.shape[0]) > 2 :
        delta = img_size[0]-img.shape[0]
        r = round(np.random.uniform(0,1), 1)
        if r <= p and p > 0.: 
            c = np.random.choice(list(range(2, delta)))
            start = np.full((c - 1, img.shape[1]), fill)
            end = np.full((delta - c, img.shape[1]), fill)
            img = np.concatenate([start, img, end], axis=0)
        else:
            half = np.full(((delta) // 2, img.shape[1]), fill)
            img = np.concatenate([half, img, half], axis=0)

    img_thrsh = cv2.threshold(img, 255 // 2, 255, cv2.THRESH_BINARY)[1]
    val, counts = np.unique(img_thrsh, return_counts=True)
    if val[counts == counts.max()][0] == 255:
        img = cv2.bitwise_not(img)
        
    img = cv2.resize(img, (img_size[1], img_size[0]), Image.LANCZOS)
    if 'name' in locals():
        return img, name.split("/")[-1].split("_")[1].lower()
    return img, False

def parse_mjsynth(path, names):
    return [os.path.join(path, name.split()[0][2:]) for name in names]

def norm(image, mean, std):
    return (image.astype('float32') - mean) / std

class Readf:

    def __init__(self, img_size=(40,40), max_len=30, normed=False, batch_size=32, classes={}, 
                 mean=118.24236953981779, std=36.72835353999682, transform_p=0.7):

        self.batch_size = batch_size
        self.transform_p = transform_p
        self.img_size = img_size
        self.normed = normed
        self.classes = classes
        self.max_len = max_len
        self.mean = mean
        self.std = std
        self.voc = list(self.classes.keys())
        if type(classes) == dict:
            self.blank = len(self.classes)

    def make_target(self, text):
        return np.array([self.classes[char] if char in self.voc else self.classes['-'] for char in text])

    def get_labels(self, names):
        Y_data = np.full([len(names), self.max_len], self.blank)
        for i, name in enumerate(names):
            img, word = open_img(name, self.img_size, p=self.transform_p)
            word = self.make_target(word)
            Y_data[i, 0:len(word)] = word
        return Y_data

    def get_blank_matrices(self):
        shape = (self.batch_size,)+self.img_size
        X_data = np.empty(shape)
        Y_data = np.full([self.batch_size, self.max_len], self.blank)
        input_length = np.ones((self.batch_size, 1))
        label_length = np.zeros((self.batch_size, 1))
        return X_data, Y_data, input_length, label_length

    def run_generator(self, names, downsample_factor=2, bboxs={}):

        if bboxs:
            n_instances = sum([len(v) for v in bboxs.values()])
        else:
            bboxs = {name:[name] for name in names}
            n_instances = len(names)

        N = n_instances // self.batch_size
        rem = n_instances % self.batch_size

        i, n = 0, 0

        source_str = []
        X_data, Y_data, input_length, label_length = self.get_blank_matrices()
            
        while True:
            for name in names:
                if bboxs[name][0] == name:
                    _img, word = open_img(name, self.img_size, p=self.transform_p)
                else:
                    img = read_img(name)

                for bbox in bboxs[name]:
                    if bbox != name:
                        _img, __ = open_img(img[bbox[1]:bbox[3], bbox[2]:bbox[4]], 
                                        self.img_size, p=self.transform_p)
                        word = bbox[0] if bbox[0] is not None else "-"

                    source_str.append(word)
                    word = self.make_target(word)
                    Y_data[i, 0:len(word)] = word
                    label_length[i] = len(word)
                    input_length[i] = (self.img_size[0]+4) // downsample_factor - 2

                    if self.normed:
                        _img = norm(_img, self.mean, self.std)

                    X_data[i] = _img[:,:,np.newaxis]
                    
                    i += 1
                    inputs = {
                        'the_input': X_data,
                        'the_labels': Y_data,
                        'input_length': input_length,
                        'label_length': label_length,
                        'source_str': np.array(source_str)
                    }
                    outputs = {'ctc': np.zeros([self.batch_size])}

                    if n == N and i == rem:
                        yield (inputs, outputs)

                    elif i == self.batch_size:
                        n += 1; i = 0
                        source_str = []
                        X_data, Y_data, input_length, label_length = self.get_blank_matrices()
                        yield (inputs, outputs)
                    
def make_ohe(y, nclasses):
    ohe = np.zeros((len(y), nclasses))
    ohe[np.arange(len(y)), y.astype('int64')] = 1
    return ohe

def get_lengths(names):
    d = {}
    for name in tqdm(names, desc="getting words lengths"):
        d[name] = len(name.split("/")[-1].split("_")[1])
    return d

def get_lexicon(non_intersecting_chars=False):
    if non_intersecting_chars:
        return list(set([i for i in '0123456789'+string.ascii_lowercase+'AaBbDdEeFfGgHhLlMmNnQqRrTt'+'-']))
    else:
        return [i for i in '0123456789'+string.ascii_lowercase+'-']

def save_model_json(model, save_path, model_name):
    model_json = model.to_json()
    with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
        json_file.write(model_json)

class EarlyStoppingIter(Callback):

    def __init__(self,
                 monitor='loss',
                 min_delta=0,
                 patience=5000,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStoppingIter, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.stopped_iter = 0
        self.restore_best_weights = restore_best_weights
        self.cycle_iterations = 0
        self.best_weights = None
        self.sum_monitor = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('\nEarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.stopped_iter = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_batch_end(self, batch, logs=None):
        self.cycle_iterations += 1
        #skip first n iters.
        logs = logs or {}
        if self.monitor not in logs:
            return

        self.sum_monitor += logs[self.monitor]

        if (self.cycle_iterations - 1) % self.patience == 0:
            current = self.sum_monitor / self.cycle_iterations

            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                if self.restore_best_weights:   
                    self.best_weights = self.model.get_weights()
            else:
                self.stopped_iter = self.cycle_iterations
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('\nRestoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_iter > 0 and self.verbose > 0:
            print('\nIteration %i: early stopping\nBest metric value: %.4f' % (self.stopped_iter + 1, self.best))
