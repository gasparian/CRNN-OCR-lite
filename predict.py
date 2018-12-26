import sys
import pickle
import time

import config_IAM as config
from imageio import imsave
from scipy import misc
import numpy as np
from PIL import Image
from keras.models import Model
from keras.models import model_from_json, load_model
from keras import backend as K
from tensorflow import ConfigProto, Session
import cv2

from reader import bbox2, set_offset_monochrome, norm, padd

def load_model(path):
    json_file = open(path+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+"/model.h5")
    return loaded_model

def init_predictor(model):
    return Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('softmax').output)

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

if __name__ == '__main__':

    path, image_path = sys.argv[1], sys.argv[2]
    gpu, pad = 0, True

    device_config = ConfigProto(intra_op_parallelism_threads=4,\
            inter_op_parallelism_threads=4, allow_soft_placement=True,\
            device_count = {'CPU' : 1, 'GPU' : gpu})
    session = Session(config=device_config)
    K.set_session(session)

    model = init_predictor(load_model(path))
    classes = pickle.load(open(path+'/classes.pickle.dat', 'rb'))
    decoder = DecodeCTCPred(model, top_paths=1, beam_width=5, 
        inverse_classes={v:k for k, v in classes.items()})

    start_time = time.time()

    img = cv2.imread(image_path)
    img = set_offset_monochrome(img, offset=config.offset, fill=config.fill)
    img = padd(img, length_bins=config.length_bins, height_bins=config.height_bins, pad=pad)

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    if img.shape[1] >= img.shape[0]:
        img = img[::-1].T
        img_size = (config.img_size[1], config.img_size[0])
        img = cv2.resize(img, img_size, Image.LANCZOS)
    else:
        img = cv2.resize(img, config.img_size, Image.LANCZOS)
    img = cv2.threshold(img, config.trsh, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    parsed_path = image_path.split('/')
    imsave('/'.join(parsed_path[:-1])+'/edited_'+parsed_path[-1], img)

    img = img[:,:,np.newaxis]
    if config.norm:
       img = norm(img)

    prediction = decoder.decode(img)
    
    print('\n', prediction, 'inference time: ',(time.time() - start_time), 'sec.\n')