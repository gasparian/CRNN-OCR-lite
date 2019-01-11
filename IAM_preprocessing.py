import pickle
import sys
from shutil import copyfile, rmtree
import re
import string
import xml.etree.ElementTree as ET
import os
import argparse

from imageio import imsave
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import padd, get_lexicon

def make_target(text, one_hot=False, classes=None):
    if one_hot:
        target = np.empty((len(text), len(classes)))
        for i, char in enumerate(text):
            ohe = np.zeros(len(classes))
            ohe[classes[char]] = 1
            target[i, :] = ohe
        return target
    else:
        voc = list(classes.keys())
        return np.array([classes[char] for char in text if char in voc])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process IAM dataset')
    parser.add_argument('-p', '--path')
    parser.add_argument('-np', '--new_path')
    args = parser.parse_args()
    globals().update(vars(args))

    """
    #lower+upper case+punct+space
    classes = {i:j for i, j in zip(string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation+' ', range(len(string.ascii_lowercase+string.ascii_uppercase+string.digits+string.punctuation)+1))}
    #lower case only
    #classes = {i:j for i, j in zip(string.ascii_lowercase+string.digits+string.punctuation+' ', range(len(string.ascii_lowercase+string.digits+string.punctuation)+1))}
    #without space symbol
    #classes = {i:j for i, j in zip(string.ascii_lowercase+string.digits+string.punctuation, range(len(string.ascii_lowercase+string.digits+string.punctuation)))}

    # punct = [i for i in string.punctuation] + ["'s", "'ve", "'ll", "'am", "'re"]
    """
    classes = {j:i for i, j in enumerate(get_lexicon())}

    # declare constants
    pad = True
    counters = [2, 16] # minimum and maximum word lengths
    height_bins = 210
    length_bins = 600

    try:
        rmtree(new_path)
    except:
        pass
    os.mkdir(new_path)

    d, target_ohe, target, lengths, c = {}, {}, {}, {}, 0
    for xml in tqdm(os.listdir(path + '/xml_data/'), desc='xml files'):
        tree = ET.parse(path+'/xml_data/'+xml)
        root = tree.getroot()
        for line in root[1]:
            line = [word for word in line if word.attrib]
            for i in range(len(line)):
                word = line[i]
                #text = word.attrib['text'].lower()
                text = word.attrib['text']
                if counters[0] <= len(text) <= counters[1]:
                    img_name = word.attrib['id'].split('-')
                    img_name = img_name[0]+'/'+'-'.join(img_name[:2])+'/'+'-'.join(img_name)+'.png'

                    img = cv2.imread(path + 'words/' + img_name)
                    try:
                        d[word.attrib['id']] = text
                        lengths[word.attrib['id']] = img.shape[1]
                    except:
                        print('file not loaded')
                        continue
                    target_ohe[word.attrib['id']] = make_target(text, one_hot=True, classes=classes)
                    target[word.attrib['id']] = make_target(text, one_hot=False, classes=classes)

                    if length_bins is None and height_bins is None:
                        copyfile(path+'words/'+img_name, path+new_path+'/'+word.attrib['id']+'.png')
                        c += 1
                        continue             

                    img = padd(img, length_bins=length_bins, height_bins=height_bins, pad=pad, left_offset=10)
                    imsave(path+new_path+'/%s.png' % word.attrib['id'], img.astype(np.uint8))
                    c += 1

    print(" [INFO] number of instances: %s" % c)
    pickle.dump(d, open(path+new_path+'/dict.pickle.dat', 'wb'))
    pickle.dump(classes, open(path+new_path+'/classes.pickle.dat', 'wb'))
    pickle.dump(lengths, open(path+new_path+'/lengths.pickle.dat', 'wb'))
    pickle.dump(target, open(path+new_path+'/target.pickle.dat', 'wb'))
    pickle.dump(target_ohe, open(path+new_path+'/target_ohe.pickle.dat', 'wb'))