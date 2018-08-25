import sys
import os
import re
import traceback
import time
import json
import pickle
from shutil import copyfile, rmtree

import tqdm
from numpy.random import RandomState
from sklearn import metrics
import numpy as np
import config_IAM as config
from sklearn.model_selection import KFold
from reader import Readf
from trainer import CnnLSTM, edit_distance, load_model_custom

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow import ConfigProto, Session

if __name__ == '__main__':

    gpu_config = ConfigProto()
    gpu_config.gpu_options.visible_device_list = '0'
    K.set_session(Session(config=gpu_config))

    path, save_path, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
    #save_path = os.path.abspath(os.path.join(path, os.pardir))

    try:
        reader = Readf(
            path, img_size=config.img_size, trsh=config.trsh, normed=config.norm, OHE=False, tl=config.tl,
            mjsynth=config.mjsynth, offset=config.offset, fill=config.fill, timesteps=config.timesteps,
            random_state=config.random_state, additional_channels=config.additional_channels,
            length_sort_mode=config.length_sort_mode, classes=config.classes
        )

        channels = 1
        if config.additional_channels:
            channels = 4
        img_size = config.img_size+(channels,)
        if config.tl:
            img_size = (config.timesteps,)+(64,64)+(channels,)

        names = np.array(list(reader.names.keys()))
        rndm = RandomState(config.random_state)
        length = len(names)

        train_indeces = rndm.choice(range(len(names)), size=int(length*config.train_portion), replace=False)
        #train = names[np.sort(train_indeces)] #sort or not? all previous results was without sorting
        train = names[train_indeces]

        val = names[[i for i in range(len(names)) if i not in train_indeces]]

        print(" [INFO] Number of classes: {}; Max. string length: {} ".format(len(reader.classes)+1, reader.max_len))

        init_model = CnnLSTM(num_classes=len(reader.classes)+1, shape=img_size, dropout=config.dropout, 
            attention=config.attention, tl=config.tl, GRU=config.GRU,
            time_dense_size=config.time_dense_size, initial_filters=config.initial_filters, 
            single_attention_vector=config.single_attention_vector,
            n_units=config.n_units, max_string_len=reader.max_len, attention_units=reader.max_len*2)
        model = init_model.get_model()
        model.summary()

        start_time = time.time()

        try:
            rmtree(save_path+"/"+model_name)
        except:
            pass
        os.mkdir(save_path+"/"+model_name)

        train_steps = len(train) // config.batch_size
        test_steps = len(val) // config.batch_size

        model_json = model.to_json()
        with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
            json_file.write(model_json)
        with open(save_path+'/'+model_name + '/model_summary.txt','w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        checkpointer = ModelCheckpoint(filepath=save_path+'/%s/checkpoint_weights.h5'%model_name, verbose=1, 
                                       save_best_only=True, save_weights_only=True)
        clr = CyclicLR(base_lr=0.001, max_lr=config.max_lr, step_size=train_steps*2, mode='triangular')

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
        pickle.dump(H.history, open(save_path+'/'+model_name+'/loss_history.pickle.dat', 'wb'))
        copyfile('./'+'config_IAM.py', save_path+'/'+model_name+'/config_IAM.py')

        #copyfile(path+'/classes.pickle.dat', save_path+'/'+model_name+'/classes.pickle.dat')
        #copyfile('/'.join(path.split('/')[:-2])+'/train_OCR.log', save_path+'/'+model_name+'/train_OCR.log')

        print(" [INFO] Computing edit distance metric with the best model...")
        model = load_model_custom(save_path+"/"+model_name, weights="checkpoint_weights")
        predicted = model.predict_generator(reader.run_generator(val, downsample_factor=2**init_model.pooling_counter),
                steps=test_steps, use_multiprocessing=False, workers=1)
        y_true = reader.get_labels(val)
        edit_distance_score = edit_distance(predicted, y_true)
        print(" [INFO] Portion of words with edit distance <= 1: %s " % edit_distance_score)

    except Exception as e:
        print(" [ERROR] "+traceback.format_exc())