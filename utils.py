import os
import re
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

def custom_load_model(path, model_name="model.json", weights="model.h5"):
    json_file = open(path+"/"+model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+"/"+weights)
    return loaded_model

class CRNN:

    def __init__(self, num_classes=97, max_string_len=23, shape=(40,40,1), time_dense_size=128, GRU=False, n_units=256, STN=False):

        self.num_classes = num_classes
        self.shape = shape
        self.max_string_len = max_string_len
        self.n_units = n_units
        self.GRU = GRU
        self.STN = STN
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
        # Spatial transformer part
        if self.STN:
            x = STN(inputs, sampling_size=self.shape[:2]) #100x32x1
            x = ZeroPadding2D(padding=(2, 2))(x) #104x36x1
        else:
            x = ZeroPadding2D(padding=(2, 2))(inputs) #104x36x1
        x = self.depthwise_conv_block(x, 64, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 128, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 256, conv_size=(3, 3), pooling=(2, 2))  #52x18x256
        x = self.depthwise_conv_block(x, 256, conv_size=(3, 3), pooling=None)
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=(1, 2))  #52x9x512
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

    def __init__(self, output_size, **kwargs):
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

class EditDistances(Callback):

    def __init__(self, inverse_classes, validation_data, validation_steps):
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.inverse_classes = inverse_classes

    def on_train_begin(self, logs={}):
        self.edit_distance = []
        self.normalized_edit_distance = []

    def on_epoch_end(self, epoch, logs={}):
        predictor = init_predictor(self.model)
        val_predict = predictor.predict_generator(self.validation_data[0], steps=self.validation_steps*2)
        val_predict = decode_predict_ctc(out=val_predict, top_paths=1, beam_width=3, inverse_classes=self.inverse_classes)
        targ = [labels_to_text(t, inverse_classes=self.inverse_classes) for t in self.validation_data[1]]
        val_edit_distance = edit_distance(val_predict, targ)
        val_normalized_edit_distance = normalized_edit_distance(val_predict, targ)
        self.edit_distance.append(val_edit_distance)
        self.normalized_edit_distance.append(val_normalized_edit_distance)
        print(" — val_edit_distance: %f — val_normalized_edit_distance %f" % (val_edit_distance, val_normalized_edit_distance))

        return

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

def decode_predict_ctc(out=None, top_paths=1, beam_width=3, inverse_classes=None):
    results = []
    greedy = False
    if top_paths == 1:
        greedy = True
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(out.shape[0]):
        text = ""
        for j in range(top_paths):
            inp = np.expand_dims(out[i], axis=0)
            labels = K.get_value(K.ctc_decode(inp, input_length=np.ones(inp.shape[0])*inp.shape[1],
                               greedy=greedy, beam_width=beam_width, top_paths=top_paths)[0][j])[0]
            text += labels_to_text(labels, inverse_classes)
        results.append(text)
    return results

class DecodeCTCPred:

    def __init__(self, predictor, top_paths=1, beam_width=5, inverse_classes=None):
        self.predictor = predictor
        self.top_paths = top_paths
        self.beam_width = beam_width
        self.inverse_classes = inverse_classes

    def labels_to_text(self, labels):
        ret = []
        for c in labels:
            if c == len(self.inverse_classes):
                ret.append("")
            else:
                ret.append(self.inverse_classes[c])
        return "".join(ret)

    def decode(self, a):
        c = np.expand_dims(a, axis=0)
        out = self.predictor.predict(c)
        results = []
        if self.beam_width < self.top_paths:
          self.beam_width = self.top_paths
        for i in range(self.top_paths):
            labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                               greedy=False, beam_width=self.beam_width, top_paths=self.top_paths)[0][i])[0]
            text = self.labels_to_text(labels)
            results.append(text)
        return results

class Readf:

    def __init__(self, path, training_file=None, img_size=(40,40), trsh=100, normed=False, fill=255, offset=5, mjsynth=False,
                    random_state=None, length_sort_mode='target', batch_size=32, classes=None, reorder=False, max_train_length=None):
        self.trsh = trsh
        self.mjsynth = mjsynth
        self.reorder = reorder
        self.path = path
        self.batch_size = batch_size
        self.img_size = img_size
        self.flag = False
        self.offset = offset
        self.fill = fill
        self.normed = normed
        if self.mjsynth:
            self.names = open(path+'/'+training_file, "r").readlines()[:max_train_length]
        else:
            self.names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
                for f in filenames if re.search('png|jpeg|jpg', f)][:max_train_length]
        self.length = len(self.names)
        self.prng = RandomState(random_state)
        self.prng.shuffle(self.names)
        self.classes = classes

        lengths = get_lengths(self.names)
        if not self.mjsynth:
            self.targets = pickle.load(open(self.path+'/target.pickle.dat', 'rb'))
            self.max_len = max([i.shape[0] for i in self.targets.values()])
        else:
            self.mean, self.std = pickle.load(open('./data/mean_std.pickle.dat', 'rb'))
            self.max_len = max(lengths.values())
        self.blank = len(self.classes)

        #reorder pictures by ascending of seq./pic length
        if self.reorder:
            if not self.mjsynth:
                if length_sort_mode == 'target':
                    self.names = OrderedDict(sorted([(k,len(self.targets[self.parse_name(k)])) for k in self.names], key=operator.itemgetter(1), reverse=False))
                elif length_sort_mode == 'shape':
                    self.names = OrderedDict(sorted([(k,lengths[self.parse_name(k)]) for k in self.names], key=operator.itemgetter(1), reverse=False))
            else:
                self.names = OrderedDict(sorted([(k,lengths[k]) for k in self.names], key=operator.itemgetter(1), reverse=False))

        print(' [INFO] Reader initialized with params: %s' % ('n_images: %i; '%self.length))

    def make_target(self, text):
        voc = list(self.classes.keys())
        return np.array([self.classes[char] if char in voc else self.classes['-'] for char in text])

    def parse_name(self, name):
        return name.split('.')[-2].split('/')[-1]

    def get_labels(self, names):
        Y_data = np.full([len(names), self.max_len], self.blank)
        c = 0
        for i, name in enumerate(names):
            if self.mjsynth:
                try:
                    img, word = self.open_img(name, pad=True)
                except:
                    continue
            else:
                word = self.targets[self.parse_name(name)]
            word = word.lower()
            c += 1
            Y_data[i, 0:len(word)] = self.make_target(word)
        return Y_data

    def open_img(self, name, pad=True):
        try:
            name = re.sub(" ", "_", name.split()[0])[1:]
        except:
            name = name[1:-1]
        img = cv2.imread(self.path+name)
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.flag:
            img_size = self.img_size
        else:
            img_size = (self.img_size[1], self.img_size[0])
        if pad and img.shape[1] < img_size[1]:
            val, counts = np.unique(img, return_counts=True)
            to_fill = val[np.where(counts == counts.max())[0][0]]
            img = np.concatenate([img, np.full((img.shape[0], img_size[1]-img.shape[1]), to_fill)], axis=1)
        img = cv2.resize(img, (img_size[1], img_size[0]), Image.LANCZOS)
        return img, name.split("_")[-2]

    def get_blank_matrices(self):
        shape = [self.batch_size, self.img_size[1], self.img_size[0], 1]
        if not self.flag:
            shape[1], shape[2] = shape[2], shape[1]

        X_data = np.empty(shape)
        Y_data = np.full([self.batch_size, self.max_len], self.blank)
        input_length = np.ones((self.batch_size, 1))
        label_length = np.zeros((self.batch_size, 1))
        return X_data, Y_data, input_length, label_length

    def run_generator(self, names, downsample_factor):
        source_str, i, n = [], 0, 0
        N = len(names) // self.batch_size
        X_data, Y_data, input_length, label_length = self.get_blank_matrices()
        while True:
            for name in names:
                if self.mjsynth:
                    try:
                        img, word = self.open_img(name, pad=True)
                    except:
                        continue

                    word = word.lower()
                    source_str.append(word)

                    word = self.make_target(word)
                    Y_data[i, 0:len(word)] = word
                    label_length[i] = len(word)

                else:
                    img = cv2.imread(name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    word = self.parse_name(name)
                    source_str.append(word)
                    word = self.targets[word]

                    Y_data[i, 0:len(word)] = word
                    label_length[i] = len(word)

                    #invert image colors.
                    img = cv2.bitwise_not(img)

                    img = cv2.resize(img, self.img_size, Image.LANCZOS)
                    img = cv2.threshold(img, self.trsh, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                if self.normed:
                    img = (img - self.mean) / self.std
                img = norm(img)

                if img.shape[1] >= img.shape[0]:
                    img = img[::-1].T
                    if not self.flag:
                        self.img_size = (self.img_size[1], self.img_size[0])
                        self.flag = True

                if self.flag:
                    input_length[i] = (self.img_size[1]+4) // downsample_factor - 2
                else:
                    input_length[i] = (self.img_size[0]+4) // downsample_factor - 2

                img = img[:,:,np.newaxis]

                X_data[i] = img
                i += 1

                if n == N and i == (len(names) % self.batch_size):
                    inputs = {
                        'the_input': X_data,
                        'the_labels': Y_data,
                        'input_length': input_length,
                        'label_length': label_length,
                        'source_str': np.array(source_str)
                    }
                    outputs = {'ctc': np.zeros([self.batch_size])}
                    yield (inputs, outputs)

                elif i == self.batch_size:
                    inputs = {
                        'the_input': X_data,
                        'the_labels': Y_data,
                        'input_length': input_length,
                        'label_length': label_length,
                        'source_str': np.array(source_str)
                    }
                    outputs = {'ctc': np.zeros([self.batch_size])}
                    source_str, i = [], 0
                    X_data, Y_data, input_length, label_length = self.get_blank_matrices()
                    n += 1
                    yield (inputs, outputs)

def get_lengths(names):
    d = {}
    for name in tqdm(names, desc="getting words lengths"):
        try:
            edited_name = re.sub(" ", "_", name.split()[0])[1:]
        except:
            edited_name = edited_name[1:-1]
        d[name] = len(edited_name.split("_")[-2])
    return d

def get_lexicon(non_intersecting_chars=False):
    if non_intersecting_chars:
        return list(set([i for i in '0123456789'+string.ascii_lowercase+'AaBbDdEeFfGgHhLlMmNnQqRrTt'+'-']))
        #return [i for i in string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation+' ']
    else:
        return [i for i in '0123456789'+string.ascii_lowercase+'-']
    

def norm(image):
    return image.astype('float32') / 255.

def dist(a, b):
    return np.power((np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2)), 1./2)

def coords2img(coords, dotSize=4, img_size=(64,64), offset=20):

    def min_max(coords):
        max_x, min_x = int(np.max(np.concatenate([coord[:, 0] for coord in coords]))), int(np.min(np.concatenate([coord[:, 0] for coord in coords]))) 
        max_y, min_y = int(np.max(np.concatenate([coord[:, 1] for coord in coords]))), int(np.min(np.concatenate([coord[:, 1] for coord in coords])))
        return min_x, max_x, min_y, max_y
    
    offset += dotSize // 2
    min_dists, dists = {}, [[] for i in range(len(coords))]
    for i, line in enumerate(coords):
        for point in line:
            dists[i].append(dist([0, 0], point))
        min_dists[min(dists[i])] = i
            
    min_dist = min(list(min_dists.keys()))
    min_index = min_dists[min_dist]
    start_point = coords[min_index][dists[min_index].index(min_dist)].copy()
    for i in range(len(coords)):
        coords[i] -= start_point
    
    min_x, max_x, min_y, max_y = min_max(coords) 
    scaleX = ((max_x - min_x) / (img_size[0]-(offset*2-1)))
    scaleY = ((max_y - min_y) / (img_size[1]-(offset*2-1)))
    for line in coords:
        line[:, 0] = line[:, 0] / scaleX
        line[:, 1] = line[:, 1] / scaleY

    min_x, max_x, min_y, max_y = min_max(coords)
        
    w = max_x-min_x+offset*2
    h = max_y-min_y+offset*2

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    start = 1
    for i in range(len(coords)):
        for j in range(len(coords[i]))[start:]:
            x, y = coords[i][j-1]
            x_n, y_n = coords[i][j]
            x -= min_x-offset; y -= min_y-offset
            x_n -= min_x-offset; y_n -= min_y-offset
            draw.line([(x,y), (x_n,y_n)], fill="black", width=dotSize)

    return {'img':img, 'scaleX':scaleX, 'scaleY':scaleY, 'start_point': start_point}

def padd(img, length_bins=[], height_bins=[], pad=True, left_offset=5):
    for axis, bins in zip([1, 0], [length_bins, height_bins]):
        if bins is not None:
            if type(bins) == int and img.shape[axis] < bins:
                if pad:
                    if axis == 1:
                        delta = bins-img.shape[1]-left_offset
                        if delta <= 0:
                            continue
                        img = np.concatenate([img, np.full((img.shape[0], delta, img.shape[-1]), 255)], axis=1)
                        img = np.concatenate([img, np.full((img.shape[0], left_offset, img.shape[-1]), 255)], axis=1)
                    else:
                        delta = bins-img.shape[0]
                        if (delta//2-1) <= 0:
                            continue
                        img = np.concatenate([np.full((delta//2-1, img.shape[1], img.shape[-1]), 255), img], axis=0)
                        img = np.concatenate([img, np.full((delta-(delta//2-1), img.shape[1], img.shape[-1]), 255)], axis=0)
                else:
                    if axis == 1:
                        img = cv2.resize(img, (bins, img.shape[0]), Image.LANCZOS)
                    else:
                        img = cv2.resize(img, (img.shape[1], bins), Image.LANCZOS)
            elif type(bins) == list:
                length = img.shape[axis]
                for j in range(len(bins)-1):
                    if bins[j] < img.shape[axis] <= ((bins[j+1] - bins[j])/2+bins[j]):
                        length = bins[j]
                    elif bins[j+1] == bins[-1] and bins[j+1] < img.shape[axis]:
                        length = bins[j+1]
                    elif ((bins[j+1] - bins[j])/2+bins[j]) < img.shape[axis] <= bins[j+1]:
                        length = bins[j+1]
                if pad:
                    if axis == 1:
                        delta = length-img.shape[1]-left_offset
                        if delta <= 0:
                            continue
                        img = np.concatenate([img, np.full((img.shape[0], delta, img.shape[-1]), 255)], axis=1)
                        img = np.concatenate([img, np.full((img.shape[0], left_offset, img.shape[-1]), 255)], axis=1)
                    else:
                        delta = length-img.shape[0]
                        if (delta//2-1) <= 0:
                            continue
                        img = np.concatenate([np.full((delta//2-1, img.shape[1], img.shape[-1]), 255), img], axis=0)
                        img = np.concatenate([img, np.full((delta-(delta//2-1), img.shape[1], img.shape[-1]), 255)], axis=0)
                else:
                    if axis == 1:
                        img = cv2.resize(img, (length, img.shape[0]), Image.LANCZOS)
                    else:
                        img = cv2.resize(img, (img.shape[1], length), Image.LANCZOS)

            elif img.shape[axis] != bins or img.shape[axis] > bins:
                continue
    return img

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = (np.where(img[rmin])[0][0], rmin)
    rmax = (np.where(img[rmax])[0][0], rmax)
    
    cmin = (cmin, np.where(img[:, cmin])[0][0])
    cmax = (cmax, np.where(img[:, cmax])[0][0])
    
    return np.array((rmin, rmax, cmin, cmax))

def set_offset_monochrome(img, offset=20, fill=0):

    shape = np.array(img.shape)
    if img.ndim > 2:
        box = bbox2(img.sum(axis=2))
    else:
        box = bbox2(img)
    box_y_min, box_y_max = np.min(box[:, 1]), np.max(box[:, 1])
    box_x_min, box_x_max = np.min(box[:, 0]), np.max(box[:, 0])
    
    y1, y2 = box_y_min - offset, box_y_max + offset
    x1, x2 = box_x_min - offset, box_x_max + offset
    
    if x1 < 0:
        x1 *= -1
        if img.ndim > 2:
            img = np.concatenate([np.full((img.shape[0], x1, img.shape[-1]), fill), img], axis=1)
        else:
            img = np.concatenate([np.full((img.shape[0], x1), fill), img], axis=1)
        x1 = 0
    if x2 > shape[1]:
        if img.ndim > 2:
            img = np.concatenate([img, np.full((img.shape[0], x2-shape[1], img.shape[-1]), fill)], axis=1)
        else:
            img = np.concatenate([img, np.full((img.shape[0], x2-shape[1]), fill)], axis=1)
        x2 = -1
    if y1 < 0:
        y1 *= -1
        if img.ndim > 2:
            img = np.concatenate([np.full((y1, img.shape[1], img.shape[-1]), fill), img], axis=0)
        else:
            img = np.concatenate([np.full((y1, img.shape[1]), fill), img], axis=0)
        y1 = 0
    if y2 > shape[0]:
        if img.ndim > 2:
            img = np.concatenate([img, np.full((y2-shape[0], img.shape[1], img.shape[-1]), fill)], axis=0)
        else:
            img = np.concatenate([img, np.full((y2-shape[0], img.shape[1]), fill)], axis=0)
        y2 = -1
        
    return img[y1:y2, x1:x2].astype(np.uint8)

def save_model_json(model, save_path, model_name):
    model_json = model.to_json()
    with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
        json_file.write(model_json)

def save_single_model(model, path):
    sinlge_model = model.layers[-2]
    sinlge_model.save(path+'/single_gpu_model.hdf5')
    model_json = sinlge_model.to_json()
    with open(path+"/single_gpu_model.json", "w") as json_file:
        json_file.write(model_json)
    sinlge_model.save_weights(path+"/single_gpu_weights.h5")

def SMA(data, dt=1, ave_t=100):
    m = data.shape[0]
    ave = int(ave_t / dt)
    data_ave = np.empty((data.shape))[ave_t:]
    for i, j in enumerate(range(ave, m)):
        data_ave[i] = np.nansum(data[(j - ave):j], axis=0) / ave
    return data_ave

def EMA(data, alpha=0.0002):
    k = 1
    length = data.shape[0]
    if data.ndim > 1: 
        av = np.empty((length, data.shape[1]))
        av[0, :] = data[0, :]
        while k < length:
            av[k, :] = av[k-1, :] + alpha * (data[k] - av[k-1, :])
            k += 1
    else:
        av = np.empty((length,)) 
        av[0] = data[0]
        while k < length:
            av[k] = av[k-1] + alpha * (data[k] - av[k-1])
            k += 1
    return av

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {"loss":[], "val_loss":[]}

    def on_batch_end(self, batch, logs={}):
        self.losses["loss"].append(logs.get('loss'))
        self.losses["val_loss"].append(logs.get('val_loss'))

class LR_Updater(Callback):

    def __init__(self):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''
        self.trn_iterations = 0.
        self.history = {}

    def setRate(self):
        return K.get_value(self.model.optimizer.lr)

    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])

    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'][n_skip:-5], self.history['loss'][n_skip:-5])
        plt.xscale('log')

class LR_Schedule(LR_Updater):

    def __init__(self, schedule={}):
        self.schedule = schedule
        self.current_lr = 0
        self.cycle_iterations = 0
        super().__init__()

    def setRate(self):
        self.cycle_iterations += 1
        for k, v in self.schedule.items():
            if self.cycle_iterations < k:
                if self.current_lr != v:
                    self.current_lr = v
                    return v
        
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={}) #changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)

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
        logs = logs or {}
        for k, v in logs.items():
            if k == self.monitor:
                self.sum_monitor += v

        if self.cycle_iterations % self.patience == 0:
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
