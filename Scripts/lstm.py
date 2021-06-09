import sys
import time
import pandas as pd
import numpy as np
import pickle
import collections
from numpy import dstack
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
def lstm(df,folder,limit,epoch=20):
    
    


    
   
    
    x, y = prepareData(df, limit)

    print("TRAINX: ", x.shape)
    
    print("TRAINY: ", y.shape)
   

    public = (y == 0).sum()
    runners = (y == 1).sum()
    print("Trajectories of public: ", public)
    print("Trajectories of runners: ", runners)
    x_train , x_test, y_train, y_test = train_test_split(x, y,test_size=0.30, random_state=42)

    publicTrain = (y_train == 0).sum()
    runnersTrain = (y_train == 1).sum()
    publicTest = (y_test == 0).sum()
    runnersTest = (y_test == 1).sum()
    print("Trajectories of public in training: ", publicTrain)
    print("Trajectories of runners in training: ", runnersTrain)
    print("Trajectories of public in testing: ", publicTest)
    print("Trajectories of runners in testing: ", runnersTest)
    print("Train: ",x_train.shape, y_train.shape)
    print("Test:",x_test.shape, y_test.shape)
    model = Sequential()
    
    model.add(LSTM(32, input_shape=(x_train.shape[1],x_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.5))
   #model.add(LSTM(32,return_sequences=False))
    #model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dropout(0.5))  
    model.add(Dense(1, activation='sigmoid'))

    name = folder
    
    checkpoint = ModelCheckpoint("models/"+name+"/"+name+"MaxValAccuracy.h5", monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=int(epoch), batch_size = 32, validation_data=(x_test,y_test), callbacks=[checkpoint])
    

    _, acc = model.evaluate(x_test, y_test)
   
    model.save("models/"+name+"/"+name+".h5")

    # predict probabilities for test set
    yhat_probs = model.predict(x_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_test, verbose=0)

        # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    #tp is the number of true positives and fp the number of false positives.
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)


   
    print("Accuracy = ", (acc * 100.0), "%")
    print("Average test loss: ", np.average(history.history['loss']))
    model.summary()
    
    
    printModel(history,name)
    
    
    
    

  
def printModel(history,name):
     #Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("./models/"+name+"/loss.png", dpi=300, bbox_inches='tight')
    plt.show()
    

    #Accuracity
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("./models/"+name+"/acc.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def prepareData(df, limit):
    limit = int(limit)
    OrientationColumnDummy = pd.get_dummies(df['Orientation'])
    df = pd.concat((df,OrientationColumnDummy),axis=1)
    df = df.drop(['Orientation'], axis=1)

    df['Type'] = df['Type'].replace(['Runner'],1)
    df['Type'] = df['Type'].replace(['Public'],0)


    videos = df['Video'].unique()

    print(df.head())
    trainX = []
    trainY = []
    first = True
    count = 2
    x_scaler  = MinMaxScaler()
    y_scaler =  MinMaxScaler() 
    for i  in videos:

        videoFrames = df.loc[df['Video'] == i]
        # print(videoFrames.head())
        ids = videoFrames['ID'].unique()
        # print("ID list: ", ids)
        for j in ids:
            
            idFrames = videoFrames.loc[df['ID']== j]
            #We don't want data from ID with less than 50 frames
            if len(idFrames) > 50: 
                #FOrmar Y
                dataY = idFrames['Type'].values
                
                idFrames = idFrames.drop(['Type','Video','ID','Frame'], axis = 1)
                dataX = idFrames.values
                #print(idFrames.info)
                #print("DataX, SIN TOCAR: ", dataX)
                #print(dataX.shape)
                if len(dataX) % limit != 0:
                    zeros = abs((len(dataX) % limit) - limit)
                  
                    zeroArray = np.zeros((zeros,11), dtype=int)
                    dataX = np.concatenate((dataX, zeroArray))
                    
                #print("DataX tras posible insercion de ceros: ", dataX)
                #print(dataX.shape)
                
                maxs = 0
                addX = []
                addY = []
                addFirst = True
                #What we do is extend our data. We use 0 to 200 then 50-250 till we got the all the data.
                #This way instead of having Limit: 200, 400 as (2,200,11) we have (5,200,11) shape.
                while maxs + limit <= len(dataX):
                    if addFirst:
                        addX = np.array(addX)
                        addY = np.array(addY)
                        addX = x_scaler.fit_transform(dataX[0+maxs:limit+maxs])
                        addY = dataY[0:1]
                        num = dataY[0]
                        addFirst = False
                    else:
                        addX = np.concatenate((addX,x_scaler.fit_transform(dataX[0+maxs:limit+maxs])))
                        addY = np.append(addY,num)
                    maxs = maxs +50
                times = int(len(addX) / limit)
                
                part  = np.reshape(addX,(times,limit,11))
                #print("PART: ",  part)
                #print("PART: ",  part.shape)
                if first:
                    trainX = np.array(trainX)
                    trainY = np.array(trainY)
                    trainX = part
                    trainY = addY
                    first = False
                else:
                    trainX = np.concatenate((trainX,part)) 
                    trainY = np.concatenate((trainY,addY))
            
            
               
            #print(idFrames)
            #print(idFrames.info())
    unique, counts = np.unique(trainY, return_counts=True)
    print("1 : Runner, 0 : Public")
    print(np.asarray((unique, counts)).T)
    trainY = trainY.reshape(len(trainY),1)
  
   
   
   
    return trainX, trainY

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
    
def loadData():
    try:
        df = pd.read_pickle('dataframe.pkl')
    except IOError:
        df = createDataFrame()  
        df.to_pickle('dataframe.pkl')
    
    return df

def loadModels(df,path_to_dir):
    x, y = prepareData(df, 400)
    print("TRAINX: ", x.shape)
    print("TRAINY: ", y.shape)
    public = (y == 0).sum()
    runners = (y == 1).sum()
    print("Trajectories of public: ", public)
    print("Trajectories of runners: ", runners)
    x_train , x_test, y_train, y_test = train_test_split(x, y,test_size=0.30, random_state=42)
    publicTrain = (y_train == 0).sum()
    runnersTrain = (y_train == 1).sum()
    publicTest = (y_test == 0).sum()
    runnersTest = (y_test == 1).sum()
    print("Trajectories of public in training: ", publicTrain)
    print("Trajectories of runners in training: ", runnersTrain)
    print("Trajectories of public in testing: ", publicTest)
    print("Trajectories of runners in testing: ", runnersTest)
    print("Train: ",x_train.shape, y_train.shape)
    print("Test:",x_test.shape, y_test.shape)
    model = load_model(path_to_dir)
    
    _, acc = model.evaluate(x_test, y_test)



    yhat_probs = model.predict(x_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_test,verbose=0)
   # print(yhat_probs)
   # print(yhat_classes)
        # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    #print(yhat_probs)
   # print(yhat_classes)
    #tp is the number of true positives and fp the number of false positives.
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test,yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)





    model.summary()
    print(model.optimizer)
    print("Accuracy = ", (acc * 100.0), "%")
    
   

def main():
    start_time = time.time()
    df = pd.DataFrame({})
    
    command = sys.argv[1]
    if command == "train":
        if len(sys.argv)>4:
            limit = sys.argv[4]
        else:
            limit = 400
        df = loadData()
        lstm(df, sys.argv[2] ,limit,sys.argv[3])
    if command == "load":
        df = loadData()
        loadModels(df,sys.argv[2])
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()