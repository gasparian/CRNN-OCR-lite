# CRNN_OCR_keras

First install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and build new image:
```
docker build -t crnn_ocr:latest -f Dockerfile .
```

Then run container in interactive mode with your paths:
```
nvidia-docker run --rm -it -v /data/OCR/data/mjsynth/mnt/ramdisk/max/90kDICT32px:/input_data -v /data/OCR/data:/save_path -p 8000:8000 crnn_ocr
```

Now you're ready to train the model:
```
python3 train.py --G 1 --path /input_data --training_fname imlist.txt --save_path /save_path --model_name CRNN_OCR_model --nbepochs 20 --norm --mjsynth --opt sgd --time_dense_size 128 --max_lr 0.002 --cyclic_lr
```

## Issues

If you have problems with the cudnn version - make changes in the first layers of image:
```
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 && \
rm -rf /var/lib/apt/lists/*
```
