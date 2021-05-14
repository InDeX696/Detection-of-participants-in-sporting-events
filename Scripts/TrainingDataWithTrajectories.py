import json
import os
import cv2
import sys
import pickle

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import collections
from collections import Counter
from sklearn import tree

from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





def trainData(df):
    backup = df
    
    df = df.drop(['Video'], axis = 1)
    
    print("------------------------------------- ")
    
  
    #Dummy variable from the categorical columns
    #This function convert categorical variables into dummy indicator variables
    
    print("------------------------------------- ")
    df['Type'] = df['Type'].replace(['Runner'],1)
    df['Type'] = df['Type'].replace(['Public'],0)
    df = df.drop(df[df.Frame < 100].index)
    print(df.describe())
    print("DataSet: ", df.info())
    y = df['Type'].values

    df = df.drop(['Type'], axis = 1)
    print(df.info())
    X = df.values
    

    print("------------------------------------- ")
    print("X values: ", X)
    print("y values: ", y)
    
    X_train , X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=0)
    a = collections.Counter(y)
    b = collections.Counter(y_train)
    c = collections.Counter(y_test)
    print("Data, 0 = Public, 1 = Runner: ",a)
    print("Data train, 0 = Public, 1 = Runner: ",b)
    print("Data test, 0 = Public, 1 = Runner: ",c)
    print("Number of tests: ",len(X_test))
    print("Number of trains: ", len(X_train))
    #DECISION TREE
    DT = decisionTree(X_train, y_train, X_test,y_test)
    #RANDOM FOREST
    RF = randomForest(X_train, y_train, X_test,y_test)
    #Gradient Boosting
    GB =gradientBoosting(X_train, y_train, X_test,y_test)
    #Naive Bayes
    NB = naiveBayes(X_train, y_train, X_test,y_test)
    #Logistic Regression
    LR =logisticRegression(X_train, y_train, X_test,y_test)
    print("------------------------------------- ")
    print("Decision Tree: ", DT)
    print("Random Forest: ", RF)
    print("Gradient Boosting: ", GB)
    print("Naive Bayes: ", NB)
    print("Logistic Regression: ", LR)
    print("------------------------------------- ")





def decisionTree(X_train, y_train, X_test,y_test):
    print("--- DECISION TREE ---")
    dt_clf = tree.DecisionTreeClassifier(max_depth=5)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    return printMetrics(y_test, y_pred)

def randomForest(X_train, y_train, X_test,y_test):
    print("\n--- RANDOM FOREST ---")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    return printMetrics(y_test, y_pred)

def gradientBoosting(X_train, y_train, X_test,y_test):
    print("\n--- Gradient Boosting ---")
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)
    return printMetrics(y_test, y_pred)

def naiveBayes(X_train, y_train, X_test,y_test):
    print("\n--- Naive Bayes---")
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    return printMetrics(y_test, y_pred)

def logisticRegression(X_train, y_train, X_test,y_test):
    print("--- Logistic Regression ---")
    lr_clf = LogisticRegression(max_iter=1000)
    print(y_train)
    lr_clf.fit(X_train, y_train)
    y_pred = lr_clf.predict(X_test)
    return printMetrics(y_test, y_pred)

def printMetrics(y_test, y_pred):
    print("------------------------------------- ")
    print("--- Confusion matrix ---")
    print(confusion_matrix(y_test,y_pred))
    print("--- Classification report ---")
    print(classification_report(y_test,y_pred))
    print("--- Accuracy score ---")
    result =accuracy_score(y_test,y_pred)
    print(result)
    return result
def createDataFrame():
    a = "DataProcessDiscretization.txt"
    with open(a, 'rb') as fp:
        data = pickle.load(fp)
    fp.close
    df = pd.DataFrame({})
    count = 0
    
    frames = 0
    for i in data:
        video = i[0]
        for j in i[1]:
            ids= j[0]
            types = j[2]
            frames = 0
            speeds = []
            height = []
            orientation = []
           
            for c in j[1]:
                frame = c[0]
                frames = frames +1
                for x in c[1]:
                    speeds.append(int(x[3]))
                    height.append(int(x[2]))
                    orientation.append(x[1])
            h = collections.Counter(height) 
            s = collections.Counter(speeds) 
            o = collections.Counter(orientation)
          
            df1 = pd.DataFrame({
                "Video": [video],
                "Frame":[frames],
                "ID": [int(ids)],
                "0":[s.get(0)],
                "1":[s.get(1)],
                "2":[s.get(2)],
                "3":[s.get(3)],
                "4":[s.get(4)],
                "5":[s.get(5)],
                "6":[s.get(6)],
                "7":[s.get(7)],
                "8":[s.get(8)],
                "9":[s.get(9)],
                "h0":[h.get(0)], 
                "h1":[h.get(1)],
                "h2":[h.get(2)],
                "h3":[h.get(3)],
                "h4":[h.get(4)],
                "h5":[h.get(5)],
                "h6":[h.get(6)],
                "h7":[h.get(7)],
                "h8":[h.get(8)],
                "h9":[h.get(9)],
                "h10":[h.get(10)],
                "North":[o.get("North")], 
                "East":[o.get("East")], 
                "West":[o.get("West")], 
                "South":[o.get("South")], 
                "NorthEast":[o.get("NorthEast")], 
                "NorthWest":[o.get("NorthWest")], 
                "SouthEast":[o.get("SouthEast")], 
                "SouthWest":[o.get("SouthWest")], 
                "Type":[types], 
            },      
                index=[count]
            )
            df = df.append(df1)
            count = count+1    
           
    df = df.fillna(value=0)   
          
    return df

   
def printData(df):
    pd.set_option('display.max_rows', None)
    print(df)
    print(df.head())
    print(df.info())
def loadData():
    df = pd.read_pickle('dataframeWithTrajectories.pkl')
    return df
def writeData(df):
    df.to_pickle('dataframeWithTrajectories.pkl')

def main():
    start_time = time.time()
    df = pd.DataFrame({})
   
    command = sys.argv[1]
    if command == "process":
        df = createDataFrame()
        writeData(df)
    elif command == "print":
        df = loadData()
        printData(df)
    elif command == "train":
        df = loadData()
        trainData(df)  

    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()