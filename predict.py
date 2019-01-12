import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
from utils import custom_load_model, init_predictor, DecodeCTCPred

if __name__ == '__main__':

    path, image_path = sys.argv[1], sys.argv[2]
    gpu, pad = 0, True

    # if len(G) == 1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = G
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device_config = ConfigProto(intra_op_parallelism_threads=4,\
            inter_op_parallelism_threads=4, allow_soft_placement=True,\
            device_count = {'CPU' : 1, 'GPU' : gpu})
    session = Session(config=device_config)
    K.set_session(session)

    model = init_predictor(custom_load_model(path))
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

    # print(" [INFO] Computing edit distance metric with the best model... ")
    # model = load_model_custom(save_path+"/"+model_name, weights="checkpoint_weights")
    # model = init_predictor(model)
    # indeces = np.random.randint(0, len(val), 10000)
    # predicted = model.predict_generator(reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h), steps=test_steps*2)
    # y_true = reader.get_labels(val)
    # true_text = [labels_to_text(y_true[i], inverse_classes=inverse_classes) for i in range(len(y_true[indeces]))]
    # predicted_text = decode_predict_ctc(out=predicted[indeces], top_paths=1, beam_width=3, inverse_classes=inverse_classes)
    # edit_distance_score = edit_distance(predicted_text, true_text)
    # normalized_edit_distance_score = normalized_edit_distance(predicted_text, true_text)
    # print(" [INFO] mean edit distance: %f ; normalized edit distance score: %f" % (edit_distance_score, normalized_edit_distance_score))