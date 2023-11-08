##Author: JAYAKRISHNA.S.S
!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()


from ultralytics import YOLO

from IPython.display import display, Image

!pip install roboflow --quiet

###copy API key from roboflow site for download the dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset = project.version(1).download("yolov8")

%cd {HOME}

!yolo task=detect mode=train model=(your_model) data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True


!ls {HOME}/runs/detect/train/

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)

#validate custom model

%cd {HOME}

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
    
#test your model
%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict3/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
