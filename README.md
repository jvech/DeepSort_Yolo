## Tutorial de instalación 


#### Instalación con conda 

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate deepsort-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate deepsort-gpu
```
#### Instalación con pip 

```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
#### Descargar pesos preentrenados y covertirlos a formato para tensorflow

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3
python load_weights.py
```



![sys](https://user-images.githubusercontent.com/50622777/118022500-0b03f200-b322-11eb-8480-1d5800b0a0fa.gif)


