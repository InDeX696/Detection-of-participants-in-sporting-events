# Deteccion-of-participants-in-sporting-events
Download YoloV4.weight and put it on Deteccion-of-participants-in-sporting-events/yolov4-deepsort/data
https://drive.google.com/file/d/1UBzWY_8ds70t2FL2-bdc9uQ6aBzPUaMU/view?usp=sharing

Download the variables.data-00000-of-00001 and put it on /yolov4-deepsort/checkpoints/yolov4-416/variables
https://drive.google.com/file/d/1cdKkeAItwSVKVqt-Amx2QFIk-U6i9PUQ/view?usp=sharing


I recommended to use conda.

You have the yml at Deteccion-of-participants-in-sporting-events\Scripts

# Tensorflow GPU
```bash
conda env create -f conda-gpu.yml
conda activate yolov4LSTM
```

# To process a video

First put your video in ./yolov4-deepsort/data/video.

From Deteccion-of-participants-in-sporting-events\yolov4-deepsort executed:

```bash
python object_tracker.py --video ./data/video/yourVideo.mp4 --output ./outputs/yourVideo-Result.avi --model yolov4 --info
```
Notice that the --info is needed.


It will create a JSON for every frame in /yolov4-deepsort/Json/VideoName

And it will output your video with the bounding boxes in ./outputs/VideoName

This program do other things but i don't use it in the project, if you want to know more please go to the original page https://github.com/theAIGuysCode/yolov4-deepsort

Next step is create a txt with the numbers of the runners on ./yolov4-deepsort/outputs/Runners

It must have the same name that the JSON folder

Now we will use the Labeller, this script will read every JSON of every person and will change the type to runner if you specify his number in the txt.

To use this script go to ./Deteccion-of-participants-in-sporting-events/Scripts

You can run this Script to tag the whole folder if you have already created a TXT file for each video.

```bash
-python .\JsonLabeller.py completefolder
```
If you want to tag only one video you can specify the name of the video's JSON folder and the name of the TXT file.

First is the JSON FOLDER second the TXT

```bash
-python .\JsonLabeller.py   tgc-parquesur-clip13 tgc-parquesur-clip13
```
Next step is calculate the speed / speed Averange / orientation and save them into a file so we can make our dataframe.

  
*All your videos must be at the same frameRate, and you must specify that frameRate.* (Our tests were conducted at 50 frames per second)

Then execute:

```bash
-python ./Scripts/JsonCharacteristic.py process NumberOfFrames
```
This will create two files, one with the calculate data that we are going to use for the LSTM(discretize data) and another one with the data without discretize.

This script can print the data in case you want to check it
```bash
-python ./Scripts/JsonCharacteristic.py print discretize.
```
You can select the video by typing the number on the left and then type *info* to see all the ID's. You can also type the ID to see his trajectory. Type *back* to navigate and *q* to exit.

The last part is training the lstm and looking at the results, but first it will create a dataframe with the data.

The script will only create the dataframe if it doesn't exist because this process can be very long if you have too many videos. So if you add more videos and you want to create
a new dataframe you have to delete the file ./Scripts/dataframe.pkl


```bash
-python lstm.py train FolderNameToSaveResults NumberOfEpochs
```
This will train the lstm and save the model in two states, when it got the best validation accuracy and in the last
epoch
SavePath : \Deteccion-of-participants-in-sporting-events\Scripts\models\FolderNameToSaveResults

This script can also load and test the lstm
```bash
-python lstm.py load ‘.\models\FolderName\FolderNameMaxValAccuracy.h5’
```


Our best result got to 97.64% of validation accuracity and 0.1846 of validation error with 2 LSTM with 64 and 32 neuron and two dense layer with 32 and 1 activation sigmoid.
<p align="center"><img src="images/model.png"\></p>
<p align="center"><img src="images/acc.png"\></p>
<p align="center"><img src="images/loss.png"\></p>