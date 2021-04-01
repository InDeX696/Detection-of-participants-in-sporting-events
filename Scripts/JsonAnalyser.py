import json
import os
import cv2
import sys
import time
import math
from os import listdir
from os.path import isfile, isdir
from collections import Counter




class dataManager:

    def __init__(self,JsonPath,ImgPath,SavePath):

        self.JsonPath = JsonPath 
        self.ImgPath = ImgPath
        self.SavePath = SavePath 
        

    def dataReader(self, flag ,id = '0',image = '') :

        PersonsDetected =[]
        idList = set()
        files = os.listdir(self.JsonPath)
        for file in files:

            with open(self.JsonPath + file) as f:
                data = json.load(f)
            for x in data["Persons"].items():

                Centroid = (int(x[1]["Centroid(W,H)"][0]),int(x[1]["Centroid(W,H)"][1]))
                if flag == "info":
                    PersonsDetected = PersonsDetected + [str(x[1]["id"])]
                elif flag == "centroidC":
                    self.centroidPrintCircle(Centroid,x[1]["x"],x[1]["y"], image,x[1]["Type"])
                elif flag == "centroidL":
                    self.centroidPrintLine(x[1]["id"],Centroid,x[1]["x"],x[1]["y"], image, x[1]["Type"],idList)
                    idList.add(x[1]["id"])

        if flag == "info":
            self.info(PersonsDetected)
        
    
    
    def centroidPrintCircle(self, Centroid, x, y, image, PersonType):
         
        if PersonType == "Runner": 
        #print(x,y,Centroid)
            image = cv2.circle(image,Centroid,0,(0,255,0),4)
        else:
            image = cv2.circle(image,Centroid,0,(0,0,255),2)

    def centroidPrintLine(self, id, Centroid, x, y, image,PersonType, idList): 
        contain = id in idList
        xmovement = []
        ymovement = []
        Pass = True
        Times = 0
        threshold = 15
        if not contain : 
            files = os.listdir(self.JsonPath)
            for file in files:

                with open(self.JsonPath + file) as f:
                    data = json.load(f)
                for x in data["Persons"].items():

                    if str(id) == str(x[1]["id"]):
                        Centroid2 = (int(x[1]["Centroid(W,H)"][0]),int(x[1]["Centroid(W,H)"][1]))
                        if Centroid2[0] != Centroid[0] or Centroid2[1] != Centroid[1]:
                            print("Frame: ",str(file))
                            print("Primer Centroide: ",Centroid, "Segundo Centroide: ",Centroid2)
                            x =abs(Centroid2[0] - Centroid[0])
                            y=abs(Centroid2[1] - Centroid[1])
                           
                            print("XDistancia: ", x, "YDistancia: ", y)
                            if Times <=9:
                                Pass = True
                                Times = Times +1
                            else:
                                SimpleMovingAverage = self.averangeCalculator(xmovement,ymovement) 
                                print("SV: ",SimpleMovingAverage)

                                if x <= (SimpleMovingAverage[0]+1)*3 and y<=(SimpleMovingAverage[1]+1)*3: 
                                    Pass = True
                                else:
                                    Pass = False
                            if Pass:

                                if PersonType == "Runner": 
                                    image = cv2.line(image, Centroid, Centroid2, (0,255,0),4) 
                                else:
                                    image = cv2.line(image, Centroid, Centroid2, (0,0,255),2) 
                                

                                if len(xmovement) >= 10:
                                    xmovement.pop(0)
                                if len(ymovement) >= 10:
                                    ymovement.pop(0)
                                xmovement.append(x)
                                ymovement.append(y)
                                print("Linea Pintada")
                                print("XArray: ",xmovement)
                                print("YArray: ",ymovement)
                            Centroid = Centroid2
                        break
    
        Centroid = 0
        Centroid2 = 0   

    def averangeCalculator(self,xmovement,ymovement):
        xsum = sum(xmovement)
        ysum = sum(ymovement)
        ax = xsum/len(xmovement)
        ay = ysum/len(ymovement)
        print("Xsum: ",xsum,"YSum: ",ysum,)
        print("AverangeX: ",ax, "AverangeY",ay)
        return math.ceil(ax),math.ceil(ay)
        
    def info(self, PersonsDetected):

        PersonsCounter = []
        RefineList = set()
        PersonsCounter = Counter(PersonsDetected)
        for i in PersonsCounter.items():
            if i[1] < 100:
                RefineList.add(i[0])
        print("----------------------------------")
        print(self.JsonPath)
        print("Number of occurrences of persons:",len(PersonsDetected))
        print("Number of occurrences of persons for every ID:",len(PersonsCounter),PersonsCounter)
        print("Ids with more than 100 occurrences:",len(RefineList),RefineList)
        print("----------------------------------")
def main():   
    start_time = time.time()
    actualPath, tail = os.path.split(os.getcwd())
    actualPath = actualPath.replace(chr(92),"/")
    actualPath = actualPath+"/yolov4-deepsort"
    Path = actualPath+"/outputs/Pictures/Originals/"

    files = os.listdir(Path)
    if sys.argv[1] == "completefolder":
        for file in files:
            FileName, FileExtension = os.path.splitext(file)
            
            

            JsonPath = actualPath + "/Json/"+FileName + "/"
            ImgPath = Path+FileName + FileExtension
            SavePath= actualPath + "/outputs/Pictures/" + FileName

            

            dataManager1 = dataManager(JsonPath,ImgPath,SavePath)

            if len(sys.argv)>1:
                command = sys.argv[2]
                if command == "info":
                    dataManager1.dataReader(flag=command)
                elif command == "centroidC":
                    SavePath = SavePath + "TrajectoryCircles.png/"
                    image = cv2.imread(ImgPath)
                    dataManager1.dataReader(image = image,flag=command)
                    cv2.imwrite(SavePath, image)
                elif command =="centroidL":
                    SavePath = SavePath + "TrajectoryLines.png/"
                    image = cv2.imread(ImgPath)
                    dataManager1.dataReader(image = image,flag=command)
                    cv2.imwrite(SavePath, image)
    else:
        FileName = sys.argv[1]
        
        JsonPath = actualPath + "/Json/"+FileName + "/"
        ImgPath = Path+FileName + ".png"
        SavePath= actualPath + "/outputs/Pictures/" + FileName

        dataManager1 = dataManager(JsonPath,ImgPath,SavePath)
        if len(sys.argv)>1:
                command = sys.argv[2]
                if command == "info":
                    dataManager1.dataReader(flag=command)
                elif command == "centroidC":
                    SavePath = SavePath + "TrajectoryCircles.png/"
                    image = cv2.imread(ImgPath)
                    dataManager1.dataReader(image = image,flag=command)
                    cv2.imwrite(SavePath, image)
                elif command =="centroidL":
                    SavePath = SavePath + "TrajectoryLines.png/"
                    image = cv2.imread(ImgPath)
                    dataManager1.dataReader(image = image,flag=command)
                    cv2.imwrite(SavePath, image)

    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()

