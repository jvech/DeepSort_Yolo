<center>
<figure>
<img  <img src="data/empty.jpeg" alt="logo" width="400"/>
<figure/>
<center/>

![LICENSE](https://img.shields.io/github/license/jmvalenciae/DeepSort_Yolo)
![GITHUB TOP LANGUAJE](https://img.shields.io/github/languages/top/jmvalenciae/DeepSort_Yolo)
![CodeFactor](https://img.shields.io/codefactor/grade/github/jmvalenciae/DeepSort_Yolo/master)
![Last commit](https://img.shields.io/github/last-commit/jmvalenciae/DeepSort_Yolo)


Tradet-Net is a software for the detection, monitoring and estimation of distances between people on videos. Its base technology is the Yolov3 deep learning model, and the DeepSort tracking algorithm.

![sys](https://user-images.githubusercontent.com/50622777/118022500-0b03f200-b322-11eb-8480-1d5800b0a0fa.gif)



## Requirements

* Python 3.9
* OS 64 bits
* RAM more or equal to 4GB



## Installation

First clone the repository

```bash
git clone --branch production https://github.com/jvech/DeepSort_Yolo.git

cd DeepSort_Yolo

python3 -m venv env

source env/bin/activate
```

```bash
# TensorFlow CPU
pip3 install -r requirements.txt

# TensorFlow GPU
pip3 install -r requirements-gpu.txt
```
## Download pretrained weights and convert them to format for tensorflow

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3
python3 load_weights.py
```

## Run

```bash
python3 main.py
```







