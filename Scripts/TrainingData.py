import json
import os
import cv2
import sys
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import collections
from collections import Counter
from sklearn import tree


def trainData(df):
    backup = df
    print("------------------------------------- ")
    print("DataSet: ", df.info())
    #Dummy variable from the categorical columns
    #This function convert categorical variables into dummy indicator variables
    OrientationColumnDummy = pd.get_dummies(df['Orientation'])
    df = pd.concat((df,OrientationColumnDummy),axis=1)
    df = df.drop(['Orientation','Video','Frame'], axis=1)

    print("------------------------------------- ")
    print("DataSet without labels and with orientation dummies", df.info())
    df['Type'] = df['Type'].replace(['Runner'],'1')
    df['Type'] = df['Type'].replace(['Public'],'0')
    y = df['Type'].values

    
    X = df.values
    X = np.delete(X,4, axis = 1)

    print("------------------------------------- ")
    print("X values: ", X)
    print("y values: ", y)
    
    
  
    X_train , X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=42)
    a = collections.Counter(y)
    b = collections.Counter(y_train)
    c = collections.Counter(y_test)
    print("Data, 0 = Public, 1 = Runner: ",a)
    print("Data train, 0 = Public, 1 = Runner: ",b)
    print("Data test, 0 = Public, 1 = Runner: ",c)
 
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
    lr_clf = LogisticRegression()
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
    
    for i in data:
        video = i[0]
        for j in i[1]:
            ids= j[0]
            types = j[2]
            for c in j[1]:
                frame = c[0]
                for x in c[1]:
                    
                    df1 = pd.DataFrame({
                        "Video": [video],
                        "Frame":[frame],
                        "ID": [int(ids)],
                        "Speed":[int(x[0])],
                        "Orientation":[x[1]],
                        "Height": [int(x[2])],
                        "SpeedAverange":[int(x[3])],
                        "Type":[types], 

                    },
                    
                    index=[count]
                    )
                    df = df.append(df1)
                    count = count+1
                    print(len(df.index))
            
    return df
def trainDataForest(df, id, video):
    df1 = pd.DataFrame({})
    df1 = df
    print("------------------------------------- ")
    print("DataSet: ", df.info())
    #Dummy variable from the categorical columns
    #This function convert categorical variables into dummy indicator variables
    OrientationColumnDummy = pd.get_dummies(df['Orientation'])
    df = pd.concat((df,OrientationColumnDummy),axis=1)
    df = df.drop(['Orientation','Video','Frame'], axis=1)

   


    print("------------------------------------- ")
    print("DataSet without labels and with orientation dummies", df.info())
    df['Type'] = df['Type'].replace(['Runner'],'1')
    df['Type'] = df['Type'].replace(['Public'],'0')
    y = df['Type'].values

    
    X = df.values
    X = np.delete(X,4, axis = 1)
    
   

    X_train , X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=0)
    
    
    
   
    
    print("\n--- RANDOM FOREST ---")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    printMetrics(y_test, y_pred)
    

    data =df1.loc[(df1['ID'] == int(id)) & (df1['Video'] == video)]
    OrientationColumnDummy = pd.get_dummies(data['Orientation'])
    data= pd.concat((data,OrientationColumnDummy),axis=1)
    data = data.drop(['Orientation','Video','Type','Frame'], axis=1)


    y_pred = rf_clf.predict(data)
    print(data)
    print(y_pred)
    a =collections.Counter(y_pred)
    print(a)
def printData(df):
    print(df)
    print(df.head())
def loadData():
    df = pd.read_pickle('dataframe.pkl')
    return df
def writeData(df):
    df.to_pickle('dataframe.pkl')

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