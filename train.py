import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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


"""
###############################################################################################
#                                       REFERENCES                                            #
###############################################################################################

https://github.com/meijieru/crnn.pytorch
https://github.com/sbillburg/CRNN-with-STN/blob/master/CRNN_with_STN.py
https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py

CTC:
https://distill.pub/2017/ctc/
https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c

Sliced RNN:
https://github.com/zepingyu0512/srnn/blob/master/SRNN.py

Attention block:
https://github.com/philipperemy/keras-attention-mechanism.git

Spatial transformer network:
https://github.com/oarriaga/STN.keras
https://arxiv.org/pdf/1506.02025.pdf
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html


###############################################################################################
#                                          LOGS:                                              #
###############################################################################################

 - casual convs: 8857k params; 16 epochs, 500k training examples: ctc_loss - ~0.91
 - depthwise separable convs and dropouts: 2785k params; 20 epochs, 500k training 
   examples; 3181s; 429ms/step; 63592 s.; ctc_loss: ~0.85

###############################################################################################

#######################################################
# TESTs section (from the inside of Deepo universal): #
#######################################################

python3 train.py --G 1 --path /data/data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px --training_fname imlist.txt \
--save_path /data/data/OCR/data --model_name OCR_mjsynth_test_TL --nbepochs 2 \
--norm --mjsynth --opt sgd --time_dense_size 128 --lr 0.002 --batch_size 64 --max_train_length 10000 \
--pretrained_path /data/data/OCR/data/OCR_mjsynth_test/single_gpu_weights.h5

########
# RUN: #
########

docker build -t crnn_ocr:latest -f Dockerfile .
nvidia-docker run --rm -it -v /data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px:/input_data -v /data/OCR/data:/save_path -p 8000:8000 crnn_ocr

python3 train.py --G 1 --path /data/data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px --training_fname annotation_train.txt \
--val_fname annotation_test.txt --save_path /data/data/OCR/data --model_name OCR_mjsynth_FULL --nbepochs 1 \
--norm --mjsynth --opt adam --time_dense_size 128 --lr .0001 --batch_size 64 --early_stopping 5000

python3 train.py --G 1 --path /data/data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px --training_fname annotation_train.txt \
--val_fname annotation_test.txt --save_path /data/data/OCR/data --model_name OCR_mjsynth_FULL_2 --nbepochs 1 \
--norm --mjsynth --opt adam --time_dense_size 128 --lr .0001 --batch_size 64 --early_stopping 20 \
--pretrained_path /data/data/OCR/data/OCR_mjsynth_FULL/checkpoint_weights.h5

____________

Mjsynth GRU
____________

python3 train.py --G 1 --path /data/data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px --training_fname annotation_train.txt \
--val_fname annotation_test.txt --save_path /data/data/OCR/data --model_name OCR_mjsynth_FULL_GRU --nbepochs 1 \
--norm --mjsynth --GRU --opt adam --time_dense_size 128 --lr .0001 --batch_size 64 --early_stopping 5000

python3 train.py --G 1 --path /data/data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px --training_fname annotation_train.txt \
--val_fname annotation_test.txt --save_path /data/data/OCR/data --model_name OCR_mjsynth_FULL_GRU_2 --nbepochs 1 \
--norm --mjsynth --GRU --opt adam --time_dense_size 128 --lr .0001 --batch_size 64 --early_stopping 20 \
--pretrained_path /data/data/OCR/data/OCR_mjsynth_FULL/checkpoint_weights.h5

____________

IAM LSTM:
____________

python3 train.py --G 1 --path /data/data/CRNN_OCR_keras/data/IAM_processed --train_portion 0.9 --save_path ./data \
--model_name OCR_IAM_ver1 --nbepochs 150 --norm --fill 255 --opt adam --time_dense_size 128 --lr .0001 \
--batch_size 64 --pretrained_path /data/data/OCR/data/OCR_mjsynth_FULL_2/final_weights.h5
____________

IAM GRU:
____________

python3 train.py --G 1 --path /data/data/CRNN_OCR_keras/data/IAM_processed --train_portion 0.9 --save_path ./data \
--model_name OCR_IAM_ver1 --nbepochs 150 --norm --fill 255 --opt adam --time_dense_size 128 --lr .0001 \
--batch_size 64 --pretrained_path /data/data/OCR/data/OCR_mjsynth_FULL_2/final_weights.h5 --GRU

##########
# TO DO: #
##########

 - make handwritten text classifier;
 - make two text box detectors: both for straight and hw text;

################
# IN PROGRESS: #
################

 - correct normalization addition;
 - retrain all with LSTM and GRU;

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='crnn_ctc_loss')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('--training_fname', type=str, required=False, default=None)
    parser.add_argument('--val_fname', type=str, required=False, default="")
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--pretrained_path', default=None, type=str, required=False)
    parser.add_argument('--nbepochs', type=int, default=20)
    parser.add_argument('--G', type=str, default="1")
    parser.add_argument('--trsh', type=int, default=100)
    parser.add_argument('--fill', type=int, default=-1)
    parser.add_argument('--offset', type=int, default=4)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--length_sort_mode', type=str, default='target')
    parser.add_argument('--train_portion', type=float, default=0.9)
    parser.add_argument('--max_train_length', type=int, default=None)
    parser.add_argument('--time_dense_size', type=int, default=128)
    parser.add_argument('--n_units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--opt', type=str, default="sgd")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--mjsynth', action='store_true')
    parser.add_argument('--GRU', action='store_true')
    parser.add_argument('--reorder', action='store_true')

    # default values set according to mjsynth dataset rules
    parser.add_argument('--imgh', type=int, default=100)
    parser.add_argument('--imgW', type=int, default=32)

    args = parser.parse_args()
    globals().update(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = G

    import tensorflow as tf
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from keras.utils.training_utils import multi_gpu_model
    from keras.models import load_model, clone_model
    from keras.layers import Lambda
    from utils import *

    try:
        rmtree(save_path+"/"+model_name)
    except:
        pass
    os.mkdir(save_path+"/"+model_name)
    with open(save_path+'/'+model_name+"/arguments.txt", "w") as f:
        f.write(str(args))

    lexicon = get_lexicon()

    classes = {j:i for i, j in enumerate(lexicon)}
    inverse_classes = {v:k for k, v in classes.items()}
    print(" [INFO] %s" % classes)

    img_size = (imgh, imgW) + (1,)
    reader = Readf(
        path, training_fname, img_size=img_size, trsh=trsh, normed=norm,
        mjsynth=mjsynth, offset=offset, fill=fill, random_state=random_state,  batch_size=batch_size, 
        length_sort_mode=length_sort_mode, classes=classes, reorder=reorder, max_train_length=max_train_length
    )

    if reorder:
        train = np.array(list(reader.names.keys()))
    else:
        train = np.array(reader.names)

    if val_fname:
        val = np.array(open(path+'/'+val_fname, "r").readlines())
    
    if not mjsynth:
        length = len(train)
        train, val = train[:int(length*train_portion)], train[int(length*train_portion):]

    print(" [INFO] Number of classes: {}; Max. string length: {} ".format(len(reader.classes)+1, reader.max_len))

    init_model = CRNN(num_classes=len(classes)+1, shape=img_size, GRU=GRU,
        time_dense_size=time_dense_size, n_units=n_units, max_string_len=reader.max_len)

    model = init_model.get_model()
    save_model_json(model, save_path, model_name)
    if pretrained_path is not None:
        model.load_weights(pretrained_path)

    train_steps = len(train) // batch_size
    if (len(train) % batch_size) > 0:
        train_steps += 1
    test_steps = len(val) // batch_size
    if (len(val) % batch_size) > 0:
        test_steps += 1

    start_time = time.time()

    with open(save_path+'/'+model_name + '/model_summary.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.summary()

    if opt == "adam":
        optimizer = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.999, clipnorm=5)
    elif opt == "sgd":
        optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(filepath=save_path+'/%s/checkpoint_weights.h5'%model_name, verbose=1, 
                                          save_best_only=True, save_weights_only=True))

    if early_stopping:
        callbacks_list.append(EarlyStoppingIter(monitor='loss', min_delta=.0001, patience=early_stopping,
                                                verbose=1, restore_best_weights=True, mode="auto"))

    H = model.fit_generator(
        generator=reader.run_generator(train, downsample_factor=2**init_model.pooling_counter_h),
        steps_per_epoch=train_steps,
        epochs=nbepochs,
        validation_data=reader.run_generator(val, downsample_factor=2**init_model.pooling_counter_h),
        validation_steps=test_steps,
        shuffle=False, verbose=1,
        callbacks=callbacks_list
    )

    pickle.dump(H.history, open(save_path+'/'+model_name+'/loss_history.pickle.dat', 'wb'))

    print(" [INFO] Training finished in %i sec.!" % (time.time() - start_time))

    model.save_weights(save_path+'/'+model_name+"/final_weights.h5")
    model.save(save_path+'/'+model_name+"/final_model.h5")

    print(" [INFO] Models and history saved! ")
