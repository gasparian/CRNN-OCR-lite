import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import argparse
import pickle
import time

from imageio import imsave
from scipy import misc
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=False, default=None)
    parser.add_argument('--G', type=int, default=-1)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--mjsynth', action='store_true')

    # default values set according to mjsynth dataset rules
    parser.add_argument('--imgh', type=int, default=100)
    parser.add_argument('--imgW', type=int, default=32)


    args = parser.parse_args()
    globals().update(vars(args))

    if not mjsynth:
        pad = True
        counters = [2, 16] # minimum and maximum word lengths
        height_bins = 210
        length_bins = 600

    if G < 0:
        from tensorflow import ConfigProto, Session
        from keras import backend as K

        device_config = ConfigProto(intra_op_parallelism_threads=4,\
                inter_op_parallelism_threads=4, allow_soft_placement=True,\
                device_count = {'CPU' : 1, 'GPU' : 0})
        session = Session(config=device_config)
        K.set_session(session)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = G

    from keras.models import Model
    from keras.models import model_from_json, load_model
    
    from reader import bbox2, set_offset_monochrome, norm, padd
    from utils import custom_load_model, init_predictor, DecodeCTCPred

    model = init_predictor(custom_load_model(model_path))
    classes = {j:i for i, j in enumerate(lexicon)}
    inverse_classes = {v:k for k, v in classes.items()}
    decoder = DecodeCTCPred(model, top_paths=1, beam_width=3, 
        inverse_classes={v:k for k, v in classes.items()})

    # add propper preprocessing

    if validate:
        # add reader object
        # fix prediction and decoding

        print(" [INFO] Computing edit distance metric with the best model... ")
        indeces = np.random.randint(0, len(val), 10000)
        predicted = model.predict_generator(reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h), steps=test_steps*2)
        y_true = reader.get_labels(val)
        true_text = [labels_to_text(y_true[i], inverse_classes=inverse_classes) for i in range(len(y_true[indeces]))]
        predicted_text = decoder.decode(out=predicted[indeces], top_paths=1, beam_width=3, inverse_classes=inverse_classes)
        edit_distance_score = edit_distance(predicted_text, true_text)
        normalized_edit_distance_score = normalized_edit_distance(predicted_text, true_text)
        print(" [INFO] mean edit distance: %f ; normalized edit distance score: %f" % (edit_distance_score, normalized_edit_distance_score))

    else:

        start_time = time.time()

        img = cv2.imread(image_path)
        img = set_offset_monochrome(img, offset=5, fill=255)
        img = padd(img, length_bins=length_bins, height_bins=height_bins, pad=pad, left_offset=5)

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