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

    # declare constants
    counters = [2, 30] # minimum and maximum word lengths

    try:
        rmtree(new_path)
    except:
        pass
    os.mkdir(new_path)

    c = 0
    for xml in tqdm(os.listdir(path + '/xml_data/'), desc='xml files'):
        tree = ET.parse(path+'/xml_data/'+xml)
        root = tree.getroot()
        for line in root[1]:
            line = [word for word in line if word.attrib]
            for i in range(len(line)):
                word = line[i]
                text = word.attrib['text'].lower()
                text = re.sub("\W+|_", " ", text)
                text = re.sub("\s+", "-", text)
                if counters[0] <= len(re.sub("-", "", text)) <= counters[1]:
                    img_name = word.attrib['id'].split('-')
                    img_name = img_name[0]+'/'+'-'.join(img_name[:2])+'/'+'-'.join(img_name)+'.png'

                    img = cv2.imread(path + '/words/' + img_name)
                    if img is None:
                        print("can't load file!")
                        continue

                    copyfile(path+'/words/'+img_name, os.path.join(new_path, "_"+text+"_"+word.attrib['id']+'.png'))
                    c += 1

    print(" [INFO] number of instances: %s" % c)
    pickle.dump(classes, open(new_path+'/classes.pickle.dat', 'wb'))