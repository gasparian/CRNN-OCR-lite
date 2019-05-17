# CRNN_OCR_lite
*Disclaimer: This is not a production-ready solution, this repo was created to just show an approach*

## Idea  
Train a light-weight network to solve word-level handwritten text recognition on images.  
## Training  
I decided to use common [CRNN](https://github.com/keras-team/keras/blob/master/examples/image_ocr.py) model with [CTC-loss](https://distill.pub/2017/ctc/) and a couple augmentations:  
- use [spatial transformer module](https://github.com/oarriaga/STN.keras) to adjust text slope;  
- replace convolution layers in CRNN with [depthwise-separable convolutions](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py);  
- use transfer learning: train OCR model on large synthetic dataset and then tune it's weights with "real" handwritten-text data.  
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/CRNN_OCR_.jpg">  

The training process consists of the following steps:  
- train model with [mjsynth](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz) dataset in two steps:  
```
python3 train.py --G 1 --path %PATH_TO_IMAGES% --training_fname annotation_train.txt \
--val_fname annotation_test.txt --save_path %NEW_PATH% --model_name %OUTPUT_MODEL_NAME% --nbepochs 1 \
--norm --mjsynth --opt adam --time_dense_size 128 --lr .0001 --batch_size 64 --early_stopping 5000

python3 train.py --G 1 --path %PATH_TO_IMAGES% --training_fname annotation_train.txt \
--val_fname annotation_test.txt --save_path %NEW_PATH% --model_name %OUTPUT_MODEL_NAME% --nbepochs 1 \
--norm --mjsynth --opt adam --time_dense_size 128 --lr .0001 --batch_size 64 --early_stopping 20 \
%PATH_TO_PRETRAINED_MODEL%/checkpoint_weights.h5
```
- prepare [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) dataset:
```
python3 IAM_preprocessing.py -p %PATH_TO_DATA% -np %PATH_TO_PROCESSED_DATA%
```
- initialize new model with weights obtained in the previous step and continue training with IAM dataset:  
```
python3 train.py --G 1 --path %PATH_TO_PROCESSED_DATA% --train_portion 0.9 --save_path %NEW_PATH% \
--model_name %OUTPUT_MODEL_NAME% --nbepochs 200 --norm --opt adam --time_dense_size 128 --lr .0001 \
--batch_size 64 --pretrained_path %PATH_TO_PRETRAINED_MODEL%/final_weights.h5
```

## Results  
After full training we've got two models: one for "reading text in the wild" and another for handwritten text transcription (you can find it in `/models`).  
I use the lowest-loss model checkpoint.  
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/IAM_losses_mjsynth.png" height=384>  

I've tested both models with random samples of 8000 images from validation sets:  
- mjsynth-model gives predictions with **.71** mean edit distance or **.09** if we normilize it by words lengths;  
- IAM-model gives **.35** mean edit distance or **.08** if we normalize it by words lengths.  

Actually, the majority of errors comes from repeated characters in true labels.  

Here are transformed images examples with transcription results:  

mjsynth | IAM  
:-------------------------:|:-------------------------:
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/mjsynth_1.png" height=225> | <img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/IAM_6.png" height=225>  
<img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/mjsynth_2.png" height=225> | <img src="https://github.com/gasparian/CRNN_OCR_lite/blob/master/imgs/STN_examples/IAM_2.png" height=225>  

For inference you can use `prediction.py` or create you own script using functions from `utils.py`:  
- mjsynth
```
python3 predict.py --G 0 --model_path %PATH_TO_MODEL% \
--image_path %PATH_TO_IMAGES% \
--val_fname annotation_test.txt --mjsynth \
--validate --num_instances 512 --max_len 23
```  
- IAM  
```
python3 predict.py --G 0 --model_path %PATH_TO_MODEL% \
--image_path %PATH_TO_IMAGES% \
--validate --num_instances 512 --max_len 21
```  
For example, this script will make prediction on images from `%PATH_TO_IMAGES%` and save results in `%PATH_TO_ANSWER%/*.csv`:   
```
python3 predict.py --G 0 --model_path %PATH_TO_MODEL% \
--image_path %PATH_TO_IMAGES% \
--result_path %PATH_TO_ANSWER% \
--max_len %MAX_STRING_LENGTH%
```  
On average, prediction on one text-box image costs us **~100-150 ms** regardless of using GPU. And **>95%** of that time consumes beam-search on LSTM output (even with fairly low beam widths: 3...10) which computes on CPU-side.  

## Reproducibility  
At first, install [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).  
Pull image from Dockerhub:
```
docker pull gasparjan/crnn_ocr:latest
```
or with CPU support only (just change tag):
```
docker pull gasparjan/crnn_ocr:cpu
```

Or build it locally:
```
docker build -t crnn_ocr:latest -f Dockerfile .
```
Run it via `nvidia-docker`, mounting volumes:
```
nvidia-docker run --rm -it -v /home:/data \
                           -p 8004:8000 gasparjan/crnn_ocr:latest
```
or just `docker` for CPU-only build:
```
docker run --rm -it -v /home:/data \
                    -p 8004:8000 gasparjan/crnn_ocr:cpu
```

...and then run scripts in shell as usual.

**The global goal is to make end-to-end pipeline for robust detection and recognition.**  

 - [x] CRNN trained on mjsynth. Training from scratch; 
 - [x] CRNN trained on IAM. Initial weights - from model trained on mjsynth; 
 - [x] CRNN trained on hand-written text "from the wild". Initial weights - from model trained on mjsynth & IAM; 
 
     - with the help of recently available [azure ocr api](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/) (check out `azure_ocr.py`) I've labeled a small dataset (148 large images) of flipcharts / whiteboards photos with a lot of handwritten text;  
     - dataset contains ~12k tokens for training and ~2k for validation. [Here is a model](https://github.com/gasparian/CRNN_OCR_lite/tree/master/models/OCR_Stickies_ver1);  
     - the results are not so convincing: **~1.6** mean edit distance and **~.3** normalized distance. To improve the recognition quality, it makes sense to apply augmentations on images / expand dataset.  
     
 - [ ] Text binarizing model (binary segmentation)
 - [ ] Word-level text boxes detector  

## P.S.  
The main usecase can be indexing recognized text on images in search: for example you've got bazillion photos of whiteboards / handwritten notes and etc. And you will be really bad at searching particullar photos with needed topic. So if the all photos had some text annotation - the problem disappears.  
Why do I think so? Clearly, it's super-hard to get 0% error rate on real-world photos. So if you want to use "hand-made" detection+recognition pipeline to "digitize" text on photos, in the end, you'll most likely need to check and correct all recognized words or add non-recognized ones. This is pretty same expirience to the current "pdf-scanners" (which is painful). And on the other side, if the model can detect and recognize even 20% of words on image, you can still find something using text search.  
