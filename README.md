 <p align="center">
  <img  <img src="data/empty.jpeg" alt="logo" width="300"/>
</p> 


  
![LICENSE](https://img.shields.io/github/license/jmvalenciae/DeepSort_Yolo)
![GITHUB TOP LANGUAJE](https://img.shields.io/github/languages/top/jmvalenciae/DeepSort_Yolo)
![CodeFactor](https://img.shields.io/codefactor/grade/github/jmvalenciae/DeepSort_Yolo/master)
![Last commit](https://img.shields.io/github/last-commit/jmvalenciae/DeepSort_Yolo)

Tradet-Net is a software for the detection, monitoring and estimation of distances between people on videos. Its base technology is the Yolov3 deep learning model, and the DeepSort tracking algorithm.

<p align="center">
  <img  <img src="https://user-images.githubusercontent.com/50622777/118022500-0b03f200-b322-11eb-8480-1d5800b0a0fa.gif" alt="logo" width="600"/>
</p> 




# Installation

## Requirements

* Python 3.9
* Operative System 64 bits
* RAM more or equal to 4GB


## Download 

Clone the repository, create virtual environment and install packages

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
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

python3 load_weights.py
```

## Run 

```bash
python3 main.py
```







