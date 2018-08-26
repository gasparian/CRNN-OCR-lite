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
from utils import *

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='quickDraw classifier')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
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
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--opt', type=str, default="adam")
    parser.add_argument('--max_lr', type=float, default=0.006)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--mjsynth', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--GRU', action='store_true')
    args = parser.parse_args()
    with open(save_path+'/'+model_name+"/arguments.txt", "w") as f:
        f.write(str(args))
    globals().update(vars(args))

    if opt == "adam":
        optimizer = optimizers.Adam(lr=0.01, beta_1=0.5, beta_2=0.999, clipnorm=5)
    elif opt == "sgd":
        optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    print("[INFO] GPU devices:%s" % get_available_gpus())

    classes = {i:j for i, j in zip(string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation+' ', range(len(string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation)+1))}
    reader = Readf(
        path, img_size=(imgh, imgW), trsh=trsh, normed=norm,
        mjsynth=mjsynth, offset=offset, fill=fill, random_state=random_state, 
        length_sort_mode=length_sort_mode, classes=classes
    )

    channels = 1
    img_size = img_size+(channels,)

    names = np.array(list(reader.names.keys()))
    rndm = RandomState(random_state)
    length = len(names)

    train_indeces = rndm.choice(range(len(names)), size=int(length*train_portion), replace=False)
    #train = names[np.sort(train_indeces)] #sort or not?
    train = names[train_indeces]
    val = names[[i for i in range(len(names)) if i not in train_indeces]]

    print(" [INFO] Number of classes: {}; Max. string length: {} ".format(len(reader.classes)+1, reader.max_len))

    init_model = CRNN(num_classes=len(reader.classes)+1, shape=img_size, dropout=dropout,
        attention=attention, GRU=GRU, time_dense_size=time_dense_size, single_attention_vector=single_attention_vector,
        n_units=n_units, max_string_len=reader.max_len, attention_units=reader.max_len*2)

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
    try:
        rmtree(save_path+"/"+model_name)
    except:
        pass
    os.mkdir(save_path+"/"+model_name)

    train_steps = len(train) // batch_size
    test_steps = len(val) // batch_size

    model_json = model.to_json()
    with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
        json_file.write(model_json)
    with open(save_path+'/'+model_name + '/model_summary.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    checkpointer = ModelCheckpoint(filepath=save_path+'/%s/checkpoint_weights.h5'%model_name, verbose=1, 
                                   save_best_only=True, save_weights_only=True)
    clr = CyclicLR(base_lr=0.001, max_lr=max_lr, step_size=train_steps*2, mode='triangular')

    H = model.fit_generator(generator=reader.run_generator(train, downsample_factor=2**init_model.pooling_counter),
                steps_per_epoch=train_steps,
                epochs=config.nbepochs,
                validation_data=reader.run_generator(val, downsample_factor=2**init_model.pooling_counter),
                validation_steps=test_steps,
                shuffle=False, verbose=1,
                callbacks=[
                    checkpointer, 
                    #clr
                ])
    print(" [INFO] Training finished in %i sec.!" % (time.time() - start_time))

    model.save_weights(save_path+'/'+model_name+"/final_weights.h5")
    model.save(save_path+'/'+model_name+"/final_model.h5")
    if G > 1:
        #save "single" model graph and weights
        save_single_model(name)
    pickle.dump(H.history, open(save_path+'/'+model_name+'/loss_history.pickle.dat', 'wb'))

    print(" [INFO] Computing edit distance metric with the best model...")
    model = load_model_custom(save_path+"/"+model_name, weights="checkpoint_weights")
    predicted = model.predict_generator(reader.run_generator(val, downsample_factor=2**init_model.pooling_counter),
            steps=test_steps, use_multiprocessing=False, workers=1)
    y_true = reader.get_labels(val)
    edit_distance_score = edit_distance(predicted, y_true)
    print(" [INFO] Portion of words with edit distance <= 1: %s " % edit_distance_score)