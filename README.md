# Deteccion-of-participants-in-sporting-events
Download YoloV4.weight and put it on Deteccion-of-participants-in-sporting-events/yolov4-deepsort/data
Download the variables.data-00000-of-00001 and put it on /yolov4-deepsort/checkpoints/yolov4-416/variables
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
First put your video in ./yolov4-deepsort/data/video
From Deteccion-of-participants-in-sporting-events\yolov4-deepsort executed:

python object_tracker.py --video ./data/video/yourVideo.mp4 --output ./outputs/yourVideo-Result.avi --model yolov4 --info

Notice that the --info is needed.


It will create a JSON for every frame in /yolov4-deepsort/Json/VideoName
And it will output your video with the bounding boxes in ./outputs/VideoName

This program do other things but i don't use it in the project, if you want to know more please go to the original page https://github.com/theAIGuysCode/yolov4-deepsort

Next step is create a txt with the numbers of the runners on ./yolov4-deepsort/outputs/Runners
It must have the same name that the JSON folder
Now we will use the Labeller, this script will read every JSON of every person and will change the type to runner
if you specify his number in the txt.

To use this script go to ./Deteccion-of-participants-in-sporting-events/Scripts

You can run this Script to tag the whole folder if you have already created a TXT file for each video.

-python .\JsonLabeller.py completefolder

If you want to tag only one video you can specify the name of the video's JSON folder and the name of the TXT file.

First is the JSON FOLDER second the TXT

-python .\JsonLabeller.py   tgc-parquesur-clip13 tgc-parquesur-clip13

Next step is calculate the speed / speed Averange / orientation and save them into a file so we can make our dataframe.

To do this you need to install a new package:

-conda install scikit-learn

  
*All your videos must be at the same frameRate, and you must specify that frameRate* Our tests were conducted at 50 frames per second.

Then execute

-python ./Scripts/JsonAnalyser.py process NumberOfFrames

This will create two files, one with the calculate data that we are going to use for the LSTM(discretize data) and another one with the data without discretize.

This script can print the data in case you want to check it
-python ./Scripts/JsonAnalyser.py print discretize.

The next step is create the panda dataframe with this data. To do this you need to install panda first.

-conda install pandas