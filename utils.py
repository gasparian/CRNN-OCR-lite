import os
import re
import pickle
import operator 
from collections import OrderedDict

import cv2
import numpy as np
from numpy.random import RandomState
from PIL import Image, ImageDraw
from tqdm import tqdm

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Input, Lambda, Bidirectional, ZeroPadding2D, concatenate, multiply
#from keras.layers import LSTM, GRU
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import CuDNNGRU as GRU
from keras.layers.core import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.models import Model, load_model, model_from_json
from keras import optimizers
from keras import backend as K

####################################################################################
#                                 REFERENCES                                       #
####################################################################################

# https://github.com/meijieru/crnn.pytorch
# https://github.com/sbillburg/CRNN-with-STN/blob/master/CRNN_with_STN.py
#
# attention block:
# https://github.com/philipperemy/keras-attention-mechanism.git

####################################################################################

class CRNN:

    def __init__(self, num_classes=97, max_string_len=23, shape=(40,40,1), attention=False, time_dense_size=128,
                       opt="adam", dropout=0.5, GRU=False, n_units=256, single_attention_vector=True):

        self.num_classes = num_classes
        self.shape = shape
        self.opt = opt
        self.attention = attention
        self.dropout = dropout
        self.max_string_len = max_string_len
        self.n_units = n_units
        self.time_dense_size = time_dense_size
        self.single_attention_vector = single_attention_vector

    def conv_block(self, inp, filters, conv_size, pooling=False, batchnorm=False, strides=None, conv_padding=(1, 1)):
        x = Conv2D(filters, conv_size, padding='valid')(inp)
        if conv_padding is not None:
            x = ZeroPadding2D(conv_padding)(x)
        if batchnorm:
            x = BatchNormalization(center=True, scale=True)(x)
        x = Activation('relu')(x)
        if pooling:
            x = MaxPooling2D((2, 2), strides=strides)(x)
            self.pooling_counter += 1
        x = Dropout(self.dropout)(x)
        return x

    def get_model(self):
        self.pooling_counter = 0
        inputs = Input(name='the_input', shape=self.shape, dtype='float32') #100x32x1
        x = ZeroPadding2D(padding=(1, 1))(inputs) #102x34x1
        x = self.conv_block(x, 64, (3, 3), pooling=False, batchnorm=False, conv_padding=(1, 1)) 
        x = self.conv_block(x, 128, (3, 3), pooling=False, batchnorm=False, conv_padding=(1, 1))
        x = self.conv_block(x, 256, (3, 3), pooling=True,  batchnorm=True, conv_padding=(1, 1)) #51x17x256
        x = self.conv_block(x, 256, (3, 3), pooling=False,  batchnorm=False, conv_padding=(1, 1))
        x = self.conv_block(x, 512, (3, 3), pooling=False,  batchnorm=True, conv_padding=(1, 1))
        x = self.conv_block(x, 512, (3, 3), pooling=False,  batchnorm=False, conv_padding=(1, 1))
        x = self.conv_block(x, 512, (2, 2),  pooling=False,  batchnorm=True, conv_padding=(1, 1)) #51x17x512

        conv_to_rnn_dims = ((self.shape[0]+2) // (2 **  self.pooling_counter), ((self.shape[1]+2) // (2 ** self.pooling_counter)) * 512) #51x8704
        x = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(x)
        x = Dense(self.time_dense_size, activation='relu', name='dense1')(x)

        if not GRU:    
            x = Bidirectional(LSTM(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum', weights=None)(x)
            x = Bidirectional(LSTM(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat', weights=None)(x)
        else:
            x = Bidirectional(GRU(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum', weights=None)(x)
            x = Bidirectional(GRU(self.n_units, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat', weights=None)(x)
        x = Dropout(self.dropout*0.5)(x)

        if self.attention:
            x = attention_3d_block(x, time_steps=conv_to_rnn_dims[0], single_attention_vector=self.single_attention_vector)

        x_ctc = Dense(self.num_classes, kernel_initializer='he_normal', name='dense2')(x)
        y_pred = Activation('softmax', name='softmax')(x_ctc)

        labels = Input(name='the_labels', shape=[self.max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        outputs = [loss_out]

        if self.opt == "adam":
            optimizer = optimizers.Adam(lr=0.01, beta_1=0.5, beta_2=0.999, clipnorm=5)
        elif self.opt == "sgd":
            optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        model = Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs)
        model.compile(optimizer=adam, loss={"ctc": lambda y_true, y_pred: y_pred})

        return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    matrix = np.zeros ((size_x, size_y))
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
    c = 0
    for y0, y in zip(y_pred, y_true):
        distance = levenshtein(y0, y)
        if distance <= 1:
            c += 1
    return c / len(y_true)

def load_model_custom(path, weights="model"):
    json_file = open(path+'/model.json', 'r')
    loaded_model_json = json_file.read()    
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+"/%s.h5" % weights)
    return loaded_model

def init_predictor(model):
    return Model(inputs=model.input, outputs=model.get_layer('softmax').output)

def labels_to_text(labels, inverse_classes=None):
    ret = []
    for c in labels:
        if c == len(inverse_classes):
            ret.append("")
        else:
            ret.append(inverse_classes[c])
    return "".join(ret)

def decode_predict_ctc(a, predictor, top_paths=1, beam_width=5):
    c = np.expand_dims(a.T, axis=0)
    out = predictor.predict(c)
    results = []
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        text = labels_to_text(labels)
        results.append(text)
    return results

class Readf:

    def __init__(self, path, img_size=(40,40), trsh=100, normed=False, fill=255, offset=5, mjsynth=False,
                    random_state=None, length_sort_mode='target', batch_size=32, classes=None):
        self.trsh = trsh
        self.mjsynth = mjsynth
        self.path = path
        self.batch_size = batch_size
        self.img_size = img_size
        self.flag = False
        self.offset = offset
        self.fill = fill
        self.normed = normed
        if self.mjsynth:
            self.names = open(path+"/imlist.txt", "r").readlines()[:500000]
        else:
            self.names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) 
                for f in filenames if re.search('png|jpeg|jpg', f)]
        self.length = len(self.names)
        self.prng = RandomState(random_state)
        self.prng.shuffle(self.names)
        self.classes = classes

        lengths = get_lengths(self.names)
        if not self.mjsynth:
            self.targets = pickle.load(open(self.path+'/target.pickle.dat', 'rb'))
            self.max_len = max([i.shape[0] for i in self.targets.values()])
        else:
            self.mean, self.std = pickle.load(open(self.path+'/mean_std.pickle.dat', 'rb'))
            self.max_len = max(lengths.values())
        self.blank = len(self.classes)

        #reorder pictures by ascending of seq./pic length
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
        return np.array([self.classes[char] for char in text if char in voc])

    def parse_name(self, name):
        return name.split('.')[-2].split('/')[-1]

    def get_labels(self, names):
        Y_data = np.full([len(names), self.max_len], self.blank)
        c = 0
        for i, name in enumerate(names):
            if self.mjsynth:
                try:
                    name = re.sub(" ", "_", name.split()[0])[1:]
                except:
                    name = name[1:-1]
                word = name.split("_")[-2]
            else:
                word = self.targets[self.parse_name(name)]
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
        if self.mjsynth:
            while True:
                for name in names:
                    try:
                        img, word = self.open_img(name, pad=True)
                    except:
                        continue

                    source_str.append(word)

                    word = self.make_target(word)
                    Y_data[i, 0:len(word)] = word
                    label_length[i] = len(word)

                    if self.normed:
                        img = (img - self.mean) / self.std
                        img = norm(img)

                    if img.shape[1] >= img.shape[0]:
                        img = img[::-1].T
                        if not self.flag:
                            self.img_size = (self.img_size[1], self.img_size[0])
                            self.flag = True

                    if self.flag:
                        input_length[i] *= (self.img_size[1]) // downsample_factor - 2
                    else:
                        input_length[i] *= (self.img_size[0]) // downsample_factor - 2

                    if not self.additional_channels:
                        img = img[:,:,np.newaxis]

                    X_data[i] = img
                    i += 1

                    if i == self.batch_size:
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


        else:
            while True:
                for name in names:

                    img = cv2.imread(name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    #no need for now, because IAM dataset already preprocessed with all needed offsets
                    #img = set_offset_monochrome(img, offset=self.offset, fill=self.fill)

                    word = self.parse_name(name)
                    source_str.append(word)
                    word = self.targets[word]

                    Y_data[i, 0:len(word)] = word
                    label_length[i] = len(word)

                    #invert image colors.
                    img = cv2.bitwise_not(img)

                    if img.shape[1] >= img.shape[0]:
                        img = img[::-1].T
                        if not self.flag:
                            self.img_size = (self.img_size[1], self.img_size[0])
                            self.flag = True

                    if self.flag:
                        input_length[i] *= (self.img_size[1]+2) // downsample_factor - 2
                    else:
                        input_length[i] *= (self.img_size[0]+2) // downsample_factor - 2

                    img = cv2.resize(img, self.img_size, Image.LANCZOS)
                    img = cv2.threshold(img, self.trsh, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    img = img[:,:,np.newaxis]

                    if self.normed:
                        img = norm(img)

                    X_data[i] = img
                    i += 1

                    if i == self.batch_size:
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

def padd(img, length_bins=[], height_bins=[], pad=True, left_offset=10):
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

#####################################################################################################################
# NOT MINE CODE:
#####################################################################################################################

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {"loss":[], "val_loss":[]}

    def on_batch_end(self, batch, logs={}):
        self.losses["loss"].append(logs.get('loss'))
        self.losses["val_loss"].append(logs.get('val_loss'))

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

#ATTENTION LAYER
def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y

def attention_3d_block(inputs, time_steps, single_attention_vector):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul') #merge is depricated
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# def model_attention_applied_after_lstm():
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     lstm_units = 32
#     lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
#     attention_mul = attention_3d_block(lstm_out)
#     attention_mul = Flatten()(attention_mul)
#     output = Dense(1, activation='sigmoid')(attention_mul)
#     model = Model(input=[inputs], output=output)
#     return model

# def model_attention_applied_before_lstm():
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     attention_mul = attention_3d_block(inputs)
#     lstm_units = 32
#     attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
#     output = Dense(1, activation='sigmoid')(attention_mul)
#     model = Model(input=[inputs], output=output)
#     return model