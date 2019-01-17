import pickle
from shutil import copyfile, rmtree
import re
import xml.etree.ElementTree as ET
import os
import argparse

import cv2
import numpy as np
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
    counters = [2, 30] # minimum and maximum word lengths

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
                if counters[0] <= len(re.sub("[^a-zA-Z0-9_]", "", text)) <= counters[1]:
                    img_name = word.attrib['id'].split('-')
                    img_name = img_name[0]+'/'+'-'.join(img_name[:2])+'/'+'-'.join(img_name)+'.png'

                    img = cv2.imread(path + '/words/' + img_name)
                    try:
                        d[word.attrib['id']] = text
                        lengths[word.attrib['id']] = img.shape[1]
                    except:
                        print("can't load file!")
                        continue
                    target[word.attrib['id']] = make_target(text)

                    copyfile(path+'/words/'+img_name, new_path+'/'+word.attrib['id']+'.png')
                    c += 1

    print(" [INFO] number of instances: %s" % c)
    pickle.dump(d, open(new_path+'/dict.pickle.dat', 'wb'))
    pickle.dump(classes, open(new_path+'/classes.pickle.dat', 'wb'))
    pickle.dump(lengths, open(new_path+'/lengths.pickle.dat', 'wb'))
    pickle.dump(target, open(new_path+'/target.pickle.dat', 'wb'))