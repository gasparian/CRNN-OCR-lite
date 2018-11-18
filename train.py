import os
import pickle
import glob
import string
import time
import math
import argparse
from shutil import copyfile, rmtree

import tqdm
import numpy as np
from numpy.random import RandomState

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model, clone_model
from keras.layers import Lambda

from utils import *

if __name__ == '__main__':

    # docker build -t crnn_ocr:latest -f Dockerfile .
    # nvidia-docker run --rm -it -v /data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px:/input_data -v /data/OCR/data:/save_path -p 8000:8000 crnn_ocr

    # python3 train.py --G 1 --path /input_data --training_fname imlist.txt --save_path /save_path --model_name CRNN_OCR_model --nbepochs 20 --norm --mjsynth --opt sgd --time_dense_size 128 --max_lr 0.002 --cyclic_lr
    # python3 train.py --G 1 --path /input_data --training_fname annotation_train.txt --val_fname annotation_test.txt --save_path /save_path --model_name OCR_ver11 --nbepochs 20 --norm --mjsynth --opt sgd --time_dense_size 128 --max_lr 0.002 --cyclic_lr

    parser = argparse.ArgumentParser(description='crnn_ctc_loss')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('--training_fname', type=str, required=True)
    parser.add_argument('--val_fname', type=str, required=False, default="")
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--nbepochs', type=int, default=20)
    parser.add_argument('--G', type=int, default=1)
    parser.add_argument('--imgh', type=int, default=100)
    parser.add_argument('--imgW', type=int, default=32)
    parser.add_argument('--trsh', type=int, default=100)
    parser.add_argument('--fill', type=int, default=255)
    parser.add_argument('--offset', type=int, default=4)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--length_sort_mode', type=str, default='target')
    parser.add_argument('--train_portion', type=float, default=0.95)
    parser.add_argument('--time_dense_size', type=int, default=128)
    parser.add_argument('--n_units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--opt', type=str, default="sgd")
    parser.add_argument('--max_lr', type=float, default=0.008)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--mjsynth', action='store_true')
    parser.add_argument('--GRU', action='store_true')
    parser.add_argument('--cyclic_lr', action='store_true')
    parser.add_argument('--reorder', action='store_true')
    parser.add_argument('--non_intersecting_chars', action='store_true')
    args = parser.parse_args()
    globals().update(vars(args))

    try:
        rmtree(save_path+"/"+model_name)
    except:
        pass
    os.mkdir(save_path+"/"+model_name)
    with open(save_path+'/'+model_name+"/arguments.txt", "w") as f:
        f.write(str(args))
    globals().update(vars(args))

    print("[INFO] GPU devices:%s" % get_available_gpus())

    if non_intersecting_chars:
        lexicon = list(set([i for i in '0123456789'+string.ascii_lowercase+'AaBbDdEeFfGgHhLlMmNnQqRrTt'+'-']))
    else:
        lexicon = [i for i in '0123456789'+string.ascii_lowercase+'-']
    #lexicon = [i for i in string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation+' ']

    classes = {j:i for i, j in enumerate(lexicon)}
    inverse_classes = {v:k for k, v in classes.items()}
    print(" [INFO] %s" % classes)

    img_size = (imgh, imgW) + (1,)
    reader = Readf(
        path, training_fname, img_size=img_size, trsh=trsh, normed=norm,
        mjsynth=mjsynth, offset=offset, fill=fill, random_state=random_state, 
        length_sort_mode=length_sort_mode, classes=classes, reorder=reorder
    )
    if reorder:
        train = np.array(list(reader.names.keys()))
    else:
        train = np.array(reader.names)

    if val_fname:
        val = np.array(open(path+'/'+val_fname, "r").readlines())

    else:
        length = len(train)
        train, val = train[:int(length*train_portion)], train[int(length*train_portion):]

    train_steps = len(train) // batch_size
    if (len(train) % batch_size) > 0:
        train_steps += 1
    test_steps = len(val) // batch_size
    if (len(val) % batch_size) > 0:
        test_steps += 1

    print(" [INFO] Number of classes: {}; Max. string length: {} ".format(len(reader.classes)+1, reader.max_len))

    init_model = CRNN(num_classes=len(classes)+1, shape=img_size, GRU=GRU, 
        time_dense_size=time_dense_size, n_units=n_units, max_string_len=reader.max_len)

    if G <= 1:
        print("[INFO] training with 1 GPU...")
        multi_model = init_model.get_model()
        model_json = multi_model.to_json()
        with open(save_path+"/"+model_name+"/model.json", "w") as json_file:
            json_file.write(model_json)
    else:
        print("[INFO] training with {} GPUs...".format(G))
     
        with tf.device("/cpu:0"):
            model = init_model.get_model()
        multi_model = multi_gpu_model(model, gpus=G)

    start_time = time.time()

    model_json = multi_model.to_json()
    with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
        json_file.write(model_json)
    with open(save_path+'/'+model_name + '/model_summary.txt','w') as f:
        multi_model.summary(print_fn=lambda x: f.write(x + '\n'))

    multi_model.summary()

    if opt == "adam":
        optimizer = optimizers.Adam(lr=max_lr, beta_1=0.5, beta_2=0.999, clipnorm=5)
    elif opt == "sgd":
        optimizer = optimizers.SGD(lr=max_lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    multi_model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    checkpointer = ModelCheckpoint(filepath=save_path+'/%s/checkpoint_weights.h5'%model_name, verbose=1, 
                               save_best_only=True, save_weights_only=True)

    if cyclic_lr:

        lr_finder = LR_Find(train_steps, min_lr=1e-5, max_lr=1, jump=2)
        multi_model.fit_generator(
            reader.run_generator(train, downsample_factor=2**init_model.pooling_counter_h),
            steps_per_epoch=train_steps,
            epochs=1, callbacks=[lr_finder]
        )

        max_lr = lr_finder.max_lr
        print("\n [INFO] Maximum learning rate: %s"%max_lr)
        lr_finder.history.update({"max_lr":max_lr})
        pickle.dump(lr_finder.history, open(save_path+'/'+model_name+'/lr_finder_history.pickle.dat', 'wb'))

        K.set_value(multi_model.optimizer.lr, max_lr)
        clr = LR_Cycle(train_steps, cycle_len=1, cycle_mult=2, epochs=nbepochs)
        H = multi_model.fit_generator(
            generator=reader.run_generator(train, downsample_factor=2**init_model.pooling_counter_h),
            steps_per_epoch=train_steps,
            epochs=nbepochs,
            validation_data=reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h),
            validation_steps=test_steps,
            shuffle=False, 
            verbose=1,
            callbacks=[checkpointer, clr])
        pickle.dump(clr.history, open(save_path+'/'+model_name+'/cycling_lr_history.pickle.dat', 'wb'))

    else:

        H = multi_model.fit_generator(
            generator=reader.run_generator(train, downsample_factor=2**init_model.pooling_counter_h),
            steps_per_epoch=train_steps,
            epochs=nbepochs,
            validation_data=reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h),
            validation_steps=test_steps,
            shuffle=False, verbose=1,
            callbacks=[checkpointer]
        )

    pickle.dump(H.history, open(save_path+'/'+model_name+'/loss_history.pickle.dat', 'wb'))

    print(" [INFO] Training finished in %i sec.!" % (time.time() - start_time))

    multi_model.save_weights(save_path+'/'+model_name+"/final_weights.h5")
    multi_model.save(save_path+'/'+model_name+"/final_model.h5")
    if G > 1:
        #save "single" model graph and weights
        save_single_model(name)
    print(" [INFO] Models and history saved! ")

    print(" [INFO] Computing edit distance metric with the best model... ")
    model = load_model_custom(save_path+"/"+model_name, weights="checkpoint_weights")
    model = init_predictor(model)
    indeces = np.random.randint(0, len(val), 10000)
    predicted = model.predict_generator(reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h), steps=test_steps*2)
    y_true = reader.get_labels(val)
    true_text = [labels_to_text(y_true[i], inverse_classes=inverse_classes) for i in range(len(y_true[indeces]))]
    predicted_text = decode_predict_ctc(out=predicted[indeces], top_paths=1, beam_width=3, inverse_classes=inverse_classes)
    edit_distance_score = edit_distance(predicted_text, true_text)
    normalized_edit_distance_score = normalized_edit_distance(predicted_text, true_text)
    print(" [INFO] mean edit distance: %f ; normalized edit distance score: %f" % (edit_distance_score, normalized_edit_distance_score))