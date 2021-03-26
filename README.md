# Deteccion-of-participants-in-sporting-events
Download YoloV4.weight and put it on Deteccion-of-participants-in-sporting-events\yolov4-deepsort\data
Getting Started with yolov4
I recommended to use conda
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu

# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt

To process one video
From Deteccion-of-participants-in-sporting-events\yolov4 executed

python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4 --info

Notice that the --info is needed.

It will create a JSON for every frame in /yolov4/Json/VideoName
And it will output your video with the bounding boxes in ./outputs/VideoName

This program do other things but i don't use it in the project, if you want to know more please go to the original page https://github.com/theAIGuysCode/yolov4-deepsort

