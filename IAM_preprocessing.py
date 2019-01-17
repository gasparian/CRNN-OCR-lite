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

from utils import get_lexicon

# python3 IAM_preprocessing.py -p ./data -np ./data/IAM_processed

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process IAM dataset')

    # in path folder should be located /words dir and /xml_data dir
    # from original IAM dataset archive
    parser.add_argument('-p', '--path') 
    parser.add_argument('-np', '--new_path')
    args = parser.parse_args()
    globals().update(vars(args))

    classes = {j:i for i, j in enumerate(get_lexicon())}

    # modified version of func. from utils
    def make_target(text):
        voc = list(classes.keys())
        return np.array([classes[char] if char in voc else classes['-'] for char in text])

    # declare constants
    resize = False
    counters = [4, 25] # minimum and maximum word lengths
    height = 210
    width = 600
    left_offset = 5

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
                text = word.attrib['text'].lower()
                #text filtering!

                img_name = word.attrib['id'].split('-')
                img_name = img_name[0]+'/'+'-'.join(img_name[:2])+'/'+'-'.join(img_name)+'.png'

                img = cv2.imread(path + '/words/' + img_name)
                try:
                    d[word.attrib['id']] = text
                    lengths[word.attrib['id']] = img.shape[1]
                except:
                    print('file not loaded')
                    continue
                target[word.attrib['id']] = make_target(text)

                if length_bins is None and height_bins is None:
                    copyfile(path+'/words/'+img_name, path+new_path+'/'+word.attrib['id']+'.png')
                    c += 1
                    continue

                if resize:
                    img, word = open_img(path+'/words/', img_name, (height, width), True, 255)
                    imsave(new_path+'/%s.png' % word.attrib['id'], img.astype(np.uint8))
                c += 1

    print(" [INFO] number of instances: %s" % c)
    pickle.dump(d, open(new_path+'/dict.pickle.dat', 'wb'))
    pickle.dump(classes, open(new_path+'/classes.pickle.dat', 'wb'))
    pickle.dump(lengths, open(new_path+'/lengths.pickle.dat', 'wb'))
    pickle.dump(target, open(new_path+'/target.pickle.dat', 'wb'))