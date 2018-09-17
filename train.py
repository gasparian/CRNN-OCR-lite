import os
import pickle
import glob
import string
import time
import argparse
from shutil import copyfile, rmtree

import tqdm
import numpy as np
from numpy.random import RandomState

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model

from utils import *

if __name__ == '__main__':

    #python3 train.py --path /data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px --save_path /data/OCR/data --model_name OCR_ver6 --nbepochs 10 --norm --mjsynth --opt sgd --time_dense_size 128

    parser = argparse.ArgumentParser(description='crnn_ctc_loss')
    parser.add_argument('-p', '--path', type=str, required=True)
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
    parser.add_argument('--single_attention_vector', type=bool, default=True)
    parser.add_argument('--n_units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--opt', type=str, default="sgd")
    parser.add_argument('--max_lr', type=float, default=0.006)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--mjsynth', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--GRU', action='store_true')
    parser.add_argument('--cyclic_RL', action='store_true')
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
    if opt == "adam":
        optimizer = optimizers.Adam(lr=0.002, beta_1=0.5, beta_2=0.999, clipnorm=5)
    elif opt == "sgd":
        optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    print("[INFO] GPU devices:%s" % get_available_gpus())

    """
    Non-intersecting letters: AaBbDdEeFfGgHhLlMmNnQqRrTt
    """
    lexicon = [i for i in '0123456789'+string.ascii_lowercase+'-']
    #lexicon = [i for i in string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation+' ']
    classes = {j:i for i, j in enumerate(lexicon)}
    inverse_classes = {v:k for k, v in classes.items()}
    print(" [INFO] %s" % classes)

    img_size = (imgh, imgW)
    reader = Readf(
        path, img_size=img_size, trsh=trsh, normed=norm,
        mjsynth=mjsynth, offset=offset, fill=fill, random_state=random_state, 
        length_sort_mode=length_sort_mode, classes=classes
    )

    img_size = img_size+(1,)
    names = np.array(list(reader.names.keys()))
    rndm = RandomState(random_state)
    length = len(names)

    train_indeces = rndm.choice(range(len(names)), size=int(length*train_portion), replace=False)
    #train = names[np.sort(train_indeces)] #sort or not?
    train = names[train_indeces]
    val = names[[i for i in range(len(names)) if i not in train_indeces]]
    train_steps = len(train) // batch_size
    if (len(train) % batch_size) > 0:
        train_steps += 1
    test_steps = len(val) // batch_size
    if (len(val) % batch_size) > 0:
        test_steps += 1

    print(" [INFO] Number of classes: {}; Max. string length: {} ".format(len(reader.classes)+1, reader.max_len))

    init_model = CRNN(num_classes=len(classes)+1, shape=img_size, attention=attention, 
        GRU=GRU, time_dense_size=time_dense_size, single_attention_vector=single_attention_vector,
        n_units=n_units, max_string_len=reader.max_len)

    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = init_model.get_model()
        model_json = model.to_json()
        with open(save_path+"/"+model_name+"/model.json", "w") as json_file:
            json_file.write(model_json)
    else:
        print("[INFO] training with {} GPUs...".format(G))
     
        with tf.device("/cpu:0"):
            model = init_model.get_model()
        multi_model = multi_gpu_model(model, gpus=G)

    model.compile(optimizer=optimizer, loss={"ctc": lambda y_true, y_pred: y_pred})
    model.summary()

    start_time = time.time()

    model_json = model.to_json()
    with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
        json_file.write(model_json)
    with open(save_path+'/'+model_name + '/model_summary.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    cb_list = []
    checkpointer = ModelCheckpoint(filepath=save_path+'/%s/checkpoint_weights.h5'%model_name, verbose=1, 
                                   save_best_only=True, save_weights_only=True)
    cb_list.append(checkpointer)
    if cyclic_RL:
        clr = CyclicLR(base_lr=0.001, max_lr=max_lr, step_size=train_steps*2, mode='triangular')
        cb_list.append(clr)

    H = model.fit_generator(generator=reader.run_generator(train, downsample_factor=2**init_model.pooling_counter_h),
                steps_per_epoch=train_steps,
                epochs=nbepochs,
                validation_data=reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h),
                validation_steps=test_steps,
                shuffle=False, verbose=1,
                callbacks=cb_list)
    print(" [INFO] Training finished in %i sec.!" % (time.time() - start_time))

    model.save_weights(save_path+'/'+model_name+"/final_weights.h5")
    model.save(save_path+'/'+model_name+"/final_model.h5")
    if G > 1:
        #save "single" model graph and weights
        save_single_model(name)
    pickle.dump(H.history, open(save_path+'/'+model_name+'/loss_history.pickle.dat', 'wb'))

    print(" [INFO] Computing edit distance metric with the best model...")
    model = load_model_custom(save_path+"/"+model_name, weights="checkpoint_weights")
    model = init_predictor(model)
    predicted = model.predict_generator(reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h), steps=test_steps*2)

    y_true = reader.get_labels(val)
    true_text = []
    for i in range(len(y_true)):
        true_text.append(labels_to_text(y_true[i], inverse_classes=inverse_classes))
    predicted_text = decode_predict_ctc(out=np.random.choice(predicted, 5000), top_paths=1, beam_width=5, inverse_classes=inverse_classes)
    edit_distance_score = edit_distance(predicted_text, true_text)
    print(" [INFO] Edit distances: %s " % (edit_distance_score))