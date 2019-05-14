import os
import re
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import argparse
import pickle
import time

from tqdm import tqdm
from scipy import misc
import numpy as np
import pandas as pd
from keras import backend as K
from numpy.random import RandomState
from PIL import Image
import cv2

"""
######################
Check mjsynth:
######################

python3 predict.py --G 0 \
                   --model_path /data/data/CRNN_OCR_keras/data/OCR_mjsynth_FULL_2 \
                   --image_path /data/data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px \
                   --val_fname annotation_test.txt --mjsynth --validate --num_instances 128 --max_len 23

######################
Check IAM:
######################

python3 predict.py --G 0 \
                   --model_path /data/data/CRNN_OCR_keras/data/OCR_IAM_ver1 \
                   --image_path /data/data/CRNN_OCR_keras/data/IAM_processed \
                   --validate --num_instances 128  --max_len 21

Predict IAM-like data and save results:

python3 predict.py --G 0 \
                   --model_path /data/data/CRNN_OCR_keras/data/OCR_IAM_ver1 \
                   --image_path /data/data/CRNN_OCR_keras/data/IAM_processed \
                   --num_instances 128 --result_path /tmp  --max_len 21

python3 predict.py --G 0 \
                   --model_path /data/data/CRNN_OCR_keras/data/OCR_IAM_ver1 \
                   --image_path /data/data/CRNN_OCR_keras/data/img \
                   --boxes /data/data/CRNN_OCR_keras/data/flipchart_words.pickle.dat \
                   --result_path /data/data/CRNN_OCR_keras/data \
                   --validate  --max_len 21

######################
Check Stickies:
######################

python3 predict.py --G 0 \
                   --model_path /data/data/CRNN_OCR_keras/data/OCR_Stickies_ver1 \
                   --image_path /data/data/CRNN_OCR_keras/data/stickies_text \
                   --train_portion 0.85 --num_instances 128 --validate  --max_len 20
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=False, default=None)
    parser.add_argument('--max_len', type=int, required=False, default=23)
    parser.add_argument('--boxes', type=str, required=False, default=None)
    parser.add_argument('--val_fname', type=str, required=False, default=None)
    parser.add_argument('--num_instances', type=int, default=None)
    parser.add_argument('--G', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--train_portion', type=float, default=.9)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--mjsynth', action='store_true')

    # default values set according to mjsynth dataset rules
    parser.add_argument('--imgh', type=int, default=100)
    parser.add_argument('--imgW', type=int, default=32)

    args = parser.parse_args()
    globals().update(vars(args))

    gpus_list = K.tensorflow_backend._get_available_gpus()
    if G < 0 or len(gpus_list) == 0:
        from tensorflow import ConfigProto, Session

        device_config = ConfigProto(
            intra_op_parallelism_threads=4,\
            inter_op_parallelism_threads=4, 
            allow_soft_placement=True,\
            device_count = {'CPU' : 1, 'GPU' : 0}
        )
        session = Session(config=device_config)
        K.set_session(session)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(G)

    from keras.models import Model
    from keras.models import model_from_json, load_model
    
    from utils import init_predictor, DecodeCTCPred, Readf, edit_distance, normalized_edit_distance, \
                        BilinearInterpolation, get_lexicon, load_custom_model, open_img, norm, parse_mjsynth

    prng = RandomState(random_state)
    model = load_custom_model(model_path, model_name='/model.json', weights="/final_weights.h5")
    model = init_predictor(model)
    classes = {j:i for i, j in enumerate(get_lexicon())}
    inverse_classes = {v:k for k, v in classes.items()}

    decoder = DecodeCTCPred(top_paths=1, beam_width=10, inverse_classes=inverse_classes)

    img_size = (imgh, imgW) + (1,)

    if validate:
        if mjsynth:
            fnames = open(os.path.join(image_path, val_fname), "r").readlines()
            fnames = np.array(parse_mjsynth(image_path, fnames))
        else:
            fnames = np.array([os.path.join(dp, f) for dp, dn, filenames in os.walk(image_path)
                                 for f in filenames if re.search('png|jpeg|jpg', f)])
            prng.shuffle(fnames)
            length = len(fnames)
            fnames = fnames[int(length*train_portion):]
    else:
        fnames = np.array([os.path.join(dp, f) for dp, dn, filenames in os.walk(image_path)
                            for f in filenames if re.search('png|jpeg|jpg', f)])

    if num_instances is not None:
        indeces = np.random.randint(0, len(fnames), min(num_instances, len(fnames)))
        fnames = fnames[indeces]

    reader = Readf(img_size=img_size, normed=True, batch_size=batch_size, 
        transform_p=0., classes=classes, max_len=max_len)
    length = len(fnames)

    bboxs = {}
    if boxes is not None:
        # get predictions from one image with detected boxes of text

        # open bboxes preprocessed file
        # data struct.: (word/None, x_0, y_0, x_1, y_1)
        bboxs = pickle.load(open(boxes, "rb"))

        # choose half of images to save time
        # and make full paths to images with boxes 
        # comment this line in "real life"
        part_dict = len(bboxs) // 2

        bboxs = {os.path.join(image_path, it[0]):it[1] for i, it in enumerate(bboxs.items()) if i <= part_dict}
        length = sum([len(v) for v in bboxs.values()])
        fnames = list(bboxs.keys())
        if validate:
            true_text = [el[0] for i, v in enumerate(bboxs.values()) 
                               for el in v if i <= part_dict]
            y_true = np.array([reader.make_target(el) for el in true_text])
    else:
        y_true = reader.get_labels(fnames)

    steps = length // batch_size
    if (length % batch_size) > 0:
        steps += 1

    print(" [INFO] Predicting... ")
    start = time.time()
    predicted = model.predict_generator(reader.run_generator(fnames, bboxs=bboxs, downsample_factor=2), steps=steps)
    print(f" [INFO] {len(fnames)} images processed in {round(time.time() - start, 2)} sec. ")

    start = time.time()
    predicted_text = decoder.decode(predicted)
    predicted_text = predicted_text[:length]
    print(f" [INFO] {len(predicted)} predictions decoded in {round(time.time() - start, 2)} sec. ")

    if result_path is not None:
        if len(fnames) != len(predicted_text):
            fnames = [fname for fname in bboxs for j in range(len(bboxs[fname]))]
        out = pd.DataFrame({"fname":fnames, "prediction":predicted_text})
        out_name = os.path.join(result_path, "prediction.csv")
        out.to_csv(out_name)
        print(" [INFO] Prediction example: \n", predicted_text[:10])
        print(" [INFO] Result store in: ", out_name)

    if validate:
        print(" [INFO] Computing edit distance metric... ")
        start = time.time()
        true_text = [decoder.labels_to_text(y_true[i]) for i in range(len(y_true))]
        print(" [INFO] Example pairs (predicted, true): \n", list(zip(predicted_text[:10], true_text[:10])))
        edit_distance_score = edit_distance(predicted_text, true_text)
        normalized_edit_distance_score = normalized_edit_distance(predicted_text, true_text)
        print(f" [INFO] edit distances calculated in {round(time.time() - start, 2)} sec. ")
        print(" [INFO] mean edit distance: %f ; normalized edit distance score: %f" % (edit_distance_score, normalized_edit_distance_score))