## Tutorial de instalación 

### En su maquina local 

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
Si está usando el sistema operativo Windows descargue los peros [aquí](https://pjreddie.com/media/files/yolov3.weights) y muevalo a la carpeta *weights*

### En la nube(Colab y Kaggle)

Para correr el sistema en la nube simplemente ejecute las siguientes lineas para preparar el entorno




```python
!git clone https://github.com/aguirrejuan/Seguimiento-de-objetivos.git
%cd /content/Seguimiento-de-objetivos/
!pip install -r requirements-gpu.txt
!wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
!python load_weights.py
```
