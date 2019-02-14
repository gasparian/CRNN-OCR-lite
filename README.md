# CRNN_OCR_lite

## Idea  
Train a light-weight network to recognize handwritten text on images.  

## Training  
I decided to use common [CRNN](https://github.com/keras-team/keras/blob/master/examples/image_ocr.py) model with [CTC-loss](https://distill.pub/2017/ctc/) and a couple augmentations:  
- use [spatial transformer module](https://github.com/oarriaga/STN.keras) to adjust text slope;  
- replace convolution layers in CRNN with [depthwise-separable convolutions](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py);
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/CRNN_OCR.jpg">  

The training process consists of the following steps:  
- train model with [mjsynth](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz) dataset;  
- initialize new model with weights obtained in the previous step and continue training with [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) dataset.  

## Results  
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/IAM_losses_mjsynth.png" height=384>  

mjsynth | IAM  
:-------------------------:|:-------------------------:
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/mjsynth_1.png" height=225> | <img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/IAM_6.png" height=225>  
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/mjsynth_2.png" height=225> | <img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/IAM_2.png" height=225>  

## Reproducibility  
```
docker build -t crnn_ocr:latest -f Dockerfile .
nvidia-docker run --rm -it -v /data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px:/input_data \
                           -v /data/OCR/data:/save_path \
                           -p 8000:8000 crnn_ocr:latest
```
