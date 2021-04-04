import json
import os
import cv2
import sys
import time
import math
import pickle
import collections
import numpy as np
from os import listdir
from os.path import isfile, isdir
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer


class dataManager:
    
    def __init__(self,JsonPath, framesPerSecond, idArray):

        self.JsonPath = JsonPath 
        self.framesPerSecond = framesPerSecond
        self.idArray = idArray
        self.idArrayD = []
        self.frameArrayD = []
        self.speedsArray = np.array([])
        self.speedsArrayA = np.array([])
    def dataReader(self):
        idList = set()
        frameArray=[]
        self.frameArrayD=[]
        files = os.listdir(self.JsonPath)
        for file in files:
            with open(self.JsonPath + file) as f:
                data = json.load(f)
            for x in data["Persons"].items():
                contain = x[1]["id"] in idList
                if not contain:
                    Centroid = (int(x[1]["Centroid(W,H)"][0]),int(x[1]["Centroid(W,H)"][1]))
                    frameArray = self.dataSearch(x[1]["id"], x[1]["x"], x[1]["y"], x[1]["height"], Centroid)   
                    idList.add(x[1]["id"])
                    self.idArray.append((x[1]["id"], frameArray,x[1]["Type"]))
                    self.idArrayD.append((x[1]["id"], self.frameArrayD,x[1]["Type"]))

        return self.idArray
    
    
    def returnD(self):
        return self.idArrayD
    def returnSpeedArray(self):
        return self.speedsArray
    
    def returnSpeedArrayA(self):
        return self.speedsArrayA   

    def dataSearch(self, id, x0, y0, height, Centroid):
        
        dataArray = []
        frameArray=[]
        dataArrayD = []
        self.frameArrayD=[]
        speedValues = []
        speedValuesFrames = []
        averange = 1
        files = os.listdir(self.JsonPath)
        for file in files:
            
            with open(self.JsonPath + file) as f:
                data = json.load(f)
            for x in data["Persons"].items():
                dataArray = []
                dataArrayD = []
                frame = str(file)
                
              
                
                if str(id) == str(x[1]["id"]):
                    Centroid2 = (int(x[1]["Centroid(W,H)"][0]),int(x[1]["Centroid(W,H)"][1]))
                    
            
                
                    h = math.sqrt(((int(Centroid2[0])-int(Centroid[0]))**2)+((int(Centroid2[1])-int(Centroid[1]))**2))
                    a = int(Centroid2[0]) - int(Centroid[0]) #x2 -x1
                    b = int(Centroid2[1]) - int(Centroid[1]) #y2-y1
                    Orientation = math.degrees(math.atan2(b,a)) #Atan ya hace y/x
                    Orientation = float("{:.2f}".format(Orientation))
                    if Orientation < 0:
                        Orientation = Orientation +360

                    Time = 2/int(self.framesPerSecond)
                    #Cambiar a centroide.
                    speed = (math.sqrt(((int(Centroid2[0])-int(Centroid[0]))**2)+((int(Centroid2[1])-int(Centroid[1]))**2)))/Time
                    speed = float("{:.2f}".format(speed))
                    boxHeight = abs(int(x[1]["height"])-int(x[1]["y"]))

                    speedValues.append(int(speed))
                    speedValuesFrames.append(frame) 
                    if len(speedValues) >5:
                        speedValues.pop(0)
                        speedValuesFrames.pop(0) 
                    if len(speedValues) <= 5:
                        speedS = sum(speedValues)
                        averange = speedS/len(speedValues)
                    aux = np.array([int(speed)])
                    aux2 = np.array([int(averange)])
                   
                    self.speedsArray = np.concatenate((self.speedsArray,aux))
                    self.speedsArrayA = np.concatenate((self.speedsArrayA,aux2))  
                    dataArray.append((speed, Orientation ,boxHeight, averange))
                    frameArray.append((frame, dataArray))

                    #Discretizacion Orientacion
                    degree = Orientation
                    degree - 22
                    if degree > 337.5  or degree <= 22.5:
                        Orientation = "North"
                    elif degree > 22.5 and degree <=67.5:
                        Orientation = "NorthEast"      
                    elif degree > 67.5 and degree <=112.5:
                        Orientation = "East"
                    elif degree > 112.5 and degree <=157.5:
                        Orientation = "SouthEast"
                    elif degree > 157.5 and degree <=202.5: 
                        Orientation = "South"
                    elif degree > 202.5 and degree <=247.5:
                        Orientation = "SouthWest"
                    elif degree > 247.5 and degree <=292.5:
                        Orientation = "West"
                    elif degree > 292.5 and degree <=337.5:
                        Orientation = "NorthWest"   
                    
                    
                    
                    #Discretizacion Velocidad
                          
                    dataArrayD.append((speed, Orientation ,boxHeight, averange))
                    self.frameArrayD.append((frame, dataArrayD))

                    Centroid = Centroid2
                    break
                
        return frameArray



def dataAnalyser(idArray, discretizer):
        dataInfo= []
        for i in idArray:
            speedJumps = 0
            heightJumps = 0
            numberOfErrors = 0
            numberOfFrames = len(i[1])
            height1 = "null"
            height2 = "null"
            speedJumpFrames = []
            heightJumpFrames = []
            times = 0
            averange = 1
            speedAverangeJumps = 0
            speedAverangeJumpFrames = []
            First = True
            
            for p in i[1]:
                #p[0] = Frame  
                flag = False       
                for q in p[1]:
                    if discretizer:
                        if int(q[0]) > 9:
                            speedJumps = speedJumps +1
                            speedJumpFrames.append(p[0])
                            flag = True
                    else:    
                        if int(q[0]) > 1000:
                            speedJumps = speedJumps +1
                            speedJumpFrames.append(p[0])
                            flag = True
 
                    if First:
                        averange = q[3]
                        First = False
                    elif averange > 1 and q[3] > averange * 2:
                        speedAverangeJumps = speedAverangeJumps +1
                        speedAverangeJumpFrames.append(p[0])
                        flag = True
                    averange = q[3]
                    if height1 == "null":
                        height1 = int(q[2])
                    else:
                        height2 = int(q[2])
                        substraction = abs(int(height2)-int(height1))
                        if substraction > height1*2 or substraction > height2*2:
                            heightJumps = heightJumps +1
                            heightJumpFrames.append(p[0])
                            flag = True
                        height1 = height2
                    if flag:
                        numberOfErrors = numberOfErrors +1  
            
            dataInfo.append((i[0], speedJumps,heightJumps,speedJumpFrames,heightJumpFrames,numberOfFrames,numberOfErrors, i[2], speedAverangeJumps,speedAverangeJumpFrames, ))
        return dataInfo
def overview(itemlist, discretizer):
    frameCount = 0
    frameErrorCount = 0
    frameCountR = 0
    frameErrorCountR = 0
    idcounter = 0
    for t in itemlist:
        
        dataInfo=dataAnalyser(t[1],discretizer)
        for d in dataInfo:
            idcounter = idcounter +1
            if d[7] == "Runner":
                frameCountR  =  frameCountR + int(d[5])
                frameErrorCountR = frameErrorCountR + int(d[6])    
            frameCount  =  frameCount + int(d[5])
            frameErrorCount = frameErrorCount + int(d[6]) 
    print("Trajectories: ", idcounter)   
    print("Total occurrences of Runners: ", frameCountR)
    print("Total occurrences of Runners with Errors: ", frameErrorCountR)
    print("Percentage of error: ", (frameErrorCountR/frameCountR)*100, "%")        
    print("Total occurrences of people: ", frameCount)
    print("Total occurrences of people with Errors: ", frameErrorCount)
    print("Percentage of error: ", (frameErrorCount/frameCount)*100, "%") 
      
  
def printData(dataType):
    speeds = []
    cardinals =[]
    discretizer = False
    if dataType == "standard":
        path = 'DataProcess.txt'
    elif dataType == "discretize":
        discretizer = True
        path = 'DataProcessDiscretization.txt'
    else:
        path = 'DataProcess.txt'
    with open (path, 'rb') as fp:
        itemlist = pickle.load(fp)
        count = 0
        print("Displaying: ", path)
        for t in itemlist:
            
            print(count,": ", t[0])
            count = count+1
        input1 = input("Enter what video do you wanna check or enter 'info' for general info: ")
        if input1 == "info":
            overview(itemlist, discretizer)
        elif input1 == "q":
            sys.exit(0)
        elif int(input1) <= len(itemlist):
            print("Selected Video: ", itemlist[int(input1)][0])
            input2 = input("Enter what Id you wanna check or enter info: ")
        
            while True :
                if input2 == "q":
                    sys.exit(0)
                if input2 == "back":
                    printData(dataType)
                if input2 == "info":
                    dataInfo = dataAnalyser(itemlist[int(input1)][1], discretizer)
                    frameCount = 0
                    frameErrorCount = 0
                    idCounter = 0
                    for d in dataInfo:
                        print("------------")
                        print("ID: ", d[0])
                        print("Type: ", d[7])
                        print("SpeedJumps: ", d[1])
                        print("HeightJumps: ", d[2])
                        print("SpeedAverangeJumps: ", d[8])
                        print("SpeedJumpFrames: ", d[3])
                        print("HeightJumpFrames: ", d[4])
                        print("SpeedAverangeFrames: ",d[9])
                        print("Frames in this trayectory: ", d[5])
                        print("Frames with Errors: ", d[6])
                        frameCount  =  frameCount + int(d[5])
                        frameErrorCount = frameErrorCount + int(d[6])
                        idCounter = idCounter +1
                    print("------------")
                    print("Total occurrences of people: ", frameCount)
                    print("Total occurrences of people with Errors: ", frameErrorCount)
                    print("Number of Trajectories; ", idCounter)
                else:   
                    flag = True 
                    speeds = []
                    speedsA = []
                    cardinals =[]
                    height=[]
                    heightCounter = collections.Counter(height)
                    speedCounterA = collections.Counter(speedsA)
                    speedCounter = collections.Counter(speeds)
                    cardinalCounter =collections.Counter(cardinals)
                    for i in itemlist[int(input1)][1]:
                        try:
                       
                            if int(i[0]) == int(input2):
                                flag = False
                                print("ID: ", i[0])
                                print("Type: ", i[2]) 
                                for p in i[1]:
                                    print("[ Frame: ", p[0]," ]")
                                    for q in p[1]:
                                        speeds.append(int(q[0]))
                                        speedsA.append(int(q[3]))
                                        height.append(int(q[2]))
                                        cardinals.append(q[1])
                                        print("Speed: ", q[0], "| Orientation: ", q[1], "| BoxHeight: ", q[2], "|SpeedAverange(5 Frames): ",q[3])
                            speedCounter = collections.Counter(speeds)
                            speedCounterA = collections.Counter(speedsA)
                            heightCounter = collections.Counter(height)
                            cardinalCounter =collections.Counter(cardinals)
                        except ValueError:
                            flag = False
                    if discretizer:
                        print("Speed Counter: ",speedCounter)
                        print("Speed Averange Counter: ", speedCounterA)
                        print("Cardinal Counter: ", cardinalCounter)
                        print("Height Counter: ", heightCounter)
                    if flag:
                        print("Id not found") 
                input2 = input("Enter what Id you wanna check or enter info: ")
def writeData(arrayData,arrayDataD):
    a = "DataProcess.txt"
    with open(a, 'wb') as fp:
        pickle.dump(arrayData, fp)
    fp.close
    a = "DataProcessDiscretization.txt"
    with open(a, 'wb') as fp:
        pickle.dump(arrayDataD, fp)
    fp.close
        
def speedDiscretization(speedsArray, arrayDataD,speedsArrayA):
    aux = np.array([0])
    #224033 +1
    if len(speedsArray) % 2 != 0:
        speedsArray=np.concatenate((speedsArray,aux))
    if len(speedsArrayA) % 2 != 0:
        speedsArrayA=np.concatenate((speedsArrayA,aux))
    speedsArrayA = speedsArrayA.reshape((int(len(speedsArrayA)/2),2))
    speedsArray = speedsArray.reshape((int(len(speedsArray)/2),2))
    est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    estA = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    est.fit(speedsArray)
    estA.fit(speedsArrayA)
    b = est.transform(speedsArray)
    bA = estA.transform(speedsArrayA)
    print(est.bin_edges_[1])
    print(est.bin_edges_[0])
    print(estA.bin_edges_[1])
    print(estA.bin_edges_[0])
    a = 0
    
    discretize = b.reshape(-1,1)
    discretizeA = bA.reshape(-1,1)
       
    arrayDataD1 = []
    frameArray1=[]
    dataArrayD1 = []
    idArrayD1 = []
    for i in arrayDataD:
        idArrayD1 = []
        frameArray1=[]
        dataArrayD1 = []
        for j in i[1]:
            frameArray1=[]
            for x in j[1]:  
                dataArrayD1 = []
                for y in x[1]:    
                  
                    if a < len(discretize)-1:
                        disHeight = int(y[2]/100)
                        dataArrayD1.append((int(discretize[a]),y[1],disHeight,int(discretizeA[a])))
                        a = a +1
                frameArray1.append((x[0],dataArrayD1))    
            idArrayD1.append((j[0],frameArray1,j[2]))       
        arrayDataD1.append((i[0],idArrayD1))                
  
    return arrayDataD1            
 
  
def main():

    start_time = time.time()
    actualPath, tail = os.path.split(os.getcwd())
    actualPath = actualPath.replace(chr(92),"/")
    pathJsons = actualPath+"/yolov4-deepsort/Json/"
    path = pathJsons
    arrayData = []
    arrayDataD = []
    speedsArray = np.array([])
    speedsArrayA = np.array([])
    if len(sys.argv)>=1:
            command = sys.argv[1]
            
            if command == "process":
                
                files = os.listdir(path)
                for file in files:
                    JsonPath = pathJsons+file+"/"
                    VideoName = file

                    
                    framesPerSecond = sys.argv[2]
                    idArray = []
                    dataManager1 = dataManager(JsonPath, framesPerSecond, idArray)

                    arrayData.append((VideoName ,dataManager1.dataReader()))
                    arrayDataD.append((VideoName ,dataManager1.returnD()))
                    aux=dataManager1.returnSpeedArray()
                    speedsArray=np.concatenate((speedsArray,aux))
                    aux2=dataManager1.returnSpeedArrayA()
                    speedsArrayA=np.concatenate((speedsArrayA,aux2))
                    
                arrayDataD=speedDiscretization(speedsArray,arrayDataD,speedsArrayA)        
                writeData(arrayData,arrayDataD)
            if command == "print":
                printData(sys.argv[2])
                   
        
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()