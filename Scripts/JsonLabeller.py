import json
import os
import sys
import time
import pathlib
from os import listdir
from os.path import isfile, isdir
from collections import Counter

class dataLabeller:

    def __init__(self,Path, GivenType, GivenIds):
        self.Path = Path
        self.GivenType = GivenType
        self.GivenIds = GivenIds 

    def dataReader(self):
        files = os.listdir(self.Path)
        for file in files:

            with open(self.Path + file) as f:
                data = json.load(f)
                f.close()
            
            for x in data["Persons"].items():
                
                for i in self.GivenIds:
                
                    if str(i) == str(x[1]["id"]):
                        
                        if str(self.GivenType) == "R":
                            x[1]["Type"] = "Runner"
                            break
                        else:
                            x[1]["Type"] = "Public"
                            break   
                    else:
                        if self.GivenType == "R":
                            x[1]["Type"] = "Public"
                        else:
                            x[1]["Type"] = "Runner"
         
            f = open(self.Path + file,"w")
            json.dump(data,f)    
            f.close()
              


def main():
    start_time = time.time()

    if len(sys.argv)>2:
        actualPath, tail = os.path.split(os.getcwd())
        actualPath = actualPath.replace(chr(92),"/")
        pathJsons = actualPath+"/yolov4-deepsort/Json/"
        nameJsonfolder = sys.argv[1]
        nameTxt = sys.argv[2]
        
        Path = pathJsons+nameJsonfolder+"/"
        LabelsPath = actualPath+"/yolov4-deepsort/outputs/Runners/" + nameTxt + ".txt"
        GivenType = "R"
        GivenIds = set({})
        f = open(LabelsPath, "r")
        i = f.read()
        if i == "all":
            GivenType = "P"
            GivenIds = set({0})
        elif i == "none":
            GivenIds = set({0})
        else:
            for n in i.split(","):
                GivenIds.add(int(n))
        print(Path, GivenType, GivenIds)  
        dataLabeller1=dataLabeller(Path, GivenType, GivenIds)
        dataLabeller1.dataReader()
    else:
        if sys.argv[1] == "completefolder":
            actualPath, tail = os.path.split(os.getcwd())
            actualPath = actualPath.replace(chr(92),"/")
            pathJsons = actualPath+"/yolov4-deepsort/Json/"
           

            files = os.listdir(pathJsons)
            for file in files:
                Path = pathJsons+file+"/"
                LabelsPath = actualPath+"/yolov4-deepsort/outputs/Runners/" + file + ".txt"
                GivenType = "R"
                GivenIds = set({})
                f = open(LabelsPath, "r")
                i = f.read()
                if i == "all":
                    GivenType = "P"
                    GivenIds = set({0})
                elif i == "none":
                    GivenIds = set({0})
                else:
                    for n in i.split(","):
                        GivenIds.add(int(n))
                print(Path, GivenType, GivenIds)  

                dataLabeller1=dataLabeller(Path, GivenType, GivenIds)
                dataLabeller1.dataReader()  
        else:
           print("Wrong command, please check readme")
    
    print("--- %s seconds ---" % (time.time() - start_time))
   

if __name__ == "__main__":
    main()
 
