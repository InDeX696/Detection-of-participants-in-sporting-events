import json
import os
import time
import sys
from os import listdir
from os.path import isfile, isdir

  
CentroidList = []
id = 2

def dataReader(path,flag = '', w = '',h='', id1 = ''):
  NumberOfFiles=0
  EmptyFiles = []
  ReliableFrames =[]
  PersonsDetected = set()
  PersonsDetectedOnReliableFrames = set()
  idList = set()
  files = os.listdir(path)
  for file in files:

    NumberOfFiles = NumberOfFiles +1 #Para info
    with open(path + file) as f:
      data = json.load(f)

    empty_check= True#Para info

    for x in data["Persons"].items():

      if flag == "info":
        empty_check = False

        PersonsDetected.add(x[1]["id"])
        if x[1]["width"] <= 1000 and x[1]["height"] >= 500:
          ReliableFrames = ReliableFrames + [str(file)]
          PersonsDetectedOnReliableFrames.add(x[1]["id"])

      elif flag == "centroidfinder":
        w1 = int(x[1]["Centroid(W,H)"][0])
        h1 = int(x[1]["Centroid(W,H)"][1])
        w = int(w)
        h = int(h)
        if w1 in range(w, w+15) or w1 in range(w-15, w):
          if h1 in range(h, h+15) or h1 in range(h-15,h):   
            idList.add(x[1]["id"])

      elif flag == "idcentroidfinder":
     
        if str(x[1]["id"]) == str(id1):
          idList.add((x[1]["id"],int(x[1]["Centroid(W,H)"][0]),int(x[1]["Centroid(W,H)"][1])))
      elif flag == "idfinder":
        if str(x[1]["id"]) == str(id1):
          print(file,"Id: ",x[1]["id"],"Type: ", x[1]["Type"],"x: ",x[1]["x"],"y: ",x[1]["y"],"width: ", x[1]["width"],"height: ",x[1]["height"],"Centroid: ",x[1]["Centroid(W,H)"])   
      elif flag =="type":
        if x[1]["Type"] == "Runner":
          idList.add(x[1]["id"])
        else:
          PersonsDetected.add(x[1]["id"])  
    if empty_check:
      EmptyFiles = EmptyFiles +[str(file)]

  if flag == "info":  
    info(EmptyFiles,ReliableFrames,NumberOfFiles,PersonsDetected,PersonsDetectedOnReliableFrames)
  elif flag == "centroidfinder":
    centroidfind(idList, w,h)
  elif flag == "idcentroidfinder":
    idfinder(idList)
  elif flag == "type":
    typeCollector(idList,PersonsDetected)



def typeCollector(idList,PersonsDetected):
  print("Runners: ", idList)
  print("Public: ", PersonsDetected)

def info(EmptyFiles,ReliableFrames,NumberOfFiles,PersonsDetected,PersonsDetectedOnReliableFrames):

  print("Numero de archivos sin personas:",len(EmptyFiles))
 # print("Numero de frames fiables(W<=1000,H>=500):",len(ReliableFrames))# ReliableFrames
  print("Numero de archivos:", NumberOfFiles)
  print("Numero de personas detectadas:",len(PersonsDetected),PersonsDetected)
  print("Numero de personas detectadas en los frames fiables:",len(PersonsDetectedOnReliableFrames),PersonsDetectedOnReliableFrames)
  
def centroidfind(idList, w, h):
  print(idList, w, h)

def idfinder(idList):
  print(idList)

def main():
  start_time = time.time()
  actualPath, tail = os.path.split(os.getcwd())
  actualPath = actualPath.replace(chr(92),"/")
  pathJsons = actualPath+"/yolov4-deepsort/Json/"
 
  if len(sys.argv)>1:
        path = pathJsons + sys.argv[1]+ "/"
        command = sys.argv[2] 
        if command == "info":
          dataReader(path,flag = command)
        if command =="centroidfinder":
          dataReader(path,flag=command,w=sys.argv[3],h=sys.argv[4])
        if command =="idcentroidfinder":
          dataReader(path,flag=command,id1 =sys.argv[3])
        if command == "idfinder":
          dataReader(path,flag=command,id1 =sys.argv[3])
        if command =="type":
          dataReader(path,flag=command)
  print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()
"""
info - informacion general de los archivos
centroidfinder - a partir de un W y H te da los ids mas cercanos +- 15
idfindercentroid- a partir de un Id te devuelve sus centroides
idfinder - a partir de un id te devuelve sus apariciones
type - Devuelve todos los ids corredores y los espectadores
"""