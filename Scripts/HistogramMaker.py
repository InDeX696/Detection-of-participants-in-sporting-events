import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn.preprocessing import KBinsDiscretizer
def main():
    a = "DataProcessDiscretization.txt"
   
    df = pd.read_pickle('dataframeWithTrajectories.pkl')
    #maxva= df['Frame'].max()
    #print(df.iloc[260,:])
   
    suma = sum(df['Frame'].values)/len(df)
    print("Frame Averange: ",suma)

    North = 0
    East = 0
    West = 0
    South = 0
    NorthEast = 0
    NorthWest = 0
    SouthEast = 0
    SouthWest = 0
    ArraySpeeds = []
    Arrayheight = []
    ArrayId = []
    speeds =[]
    ArrayVideo = []
    df1 = df['Frame'].values
    df = df['Frame'].values
    df = df[df > 50]
    print(df)
    
    if len(df)%2 != 0: 
        df = df[:-1]
    df = df.reshape((int(len(df)/2),2))
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    est.fit(df)
    b = est.transform(df)
    df = b.reshape(-1,1)
    dfi = est.bin_edges_[1]
    dfIntervals = []
    for i in dfi:
        dfIntervals.append('{:.4f}'.format(i))
    
    with open (a, 'rb') as fp:
        itemlist = pickle.load(fp) 
        for t in itemlist:

            for i in t[1]:
                speeds =[]
                for j in i[1]:
                    for x in j[1]:
                        if x[1] == "North":
                            North = North +1
                        elif x[1] == "East":
                            East = East +1
                        elif x[1] == "West":
                            West = West +1
                        elif x[1]  == "South":
                            South = South+1
                        elif x[1] == "NorthEast":
                            NorthEast = NorthEast +1
                        elif  x[1] == "NorthWest":
                            NorthWest = NorthWest +1
                        elif x[1] == "SouthEast":
                            SouthEast = SouthEast+1
                        elif x[1] == "SouthWest":
                            SouthWest = SouthWest+1
                        Arrayheight.append(int(x[2]))
                        ArraySpeeds.append(int(x[3]))
                        
                  
            ArrayId.append((i[0],speeds))
        ArrayVideo.append((t[0],ArrayId))
   # print(ArrayVideo)
    #print(North, East,West,South ,NorthEast ,NorthWest ,SouthEast ,SouthWest)
    a = [North, East,West,South ,NorthEast ,NorthWest ,SouthEast ,SouthWest]
    speeds = collections.Counter(ArraySpeeds)    
    heigh = collections.Counter(Arrayheight)   
    #print(speeds)
    #print(heigh)
    Total = sum(a)
    Labels =['North', 'East','West','South' ,'NorthEast','NorthWest' ,'SouthEast' ,'SouthWest'] 
    y_pos = np.arange(len(Labels))
    plt.bar(y_pos, a, color = (0.5,0.1,0.5,0.6))

    plt.title('Cardinal direction')
    plt.xlabel('Cardinals')
    plt.ylabel('Times')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(y_pos, Labels)
    plt.show()

    
    intervalos = [0,1,2,3,4,5,6,7,8,9,10,11]
    
    for i in intervalos:
        if i < len(dfIntervals):
            print(dfIntervals[i], i)
    plt.hist(x=df, bins=intervalos, color='#F2AB6D', rwidth=0.85)
    plt.title('Frames Histogram')
    plt.xlabel('Frames')
    plt.ylabel('Times')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(intervalos) 
    plt.show()



    intervalos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    print(len(ArraySpeeds))
    plt.hist(x=ArraySpeeds, bins=intervalos, color='#F2AB6D', rwidth=0.85)
    plt.title('Speed Histogram')
    plt.xlabel('Speed')
    plt.ylabel('Times')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(intervalos) 
    plt.show()

    intervalos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    print(len(Arrayheight))
    plt.hist(x=Arrayheight, bins=intervalos, color='#F2AB6D', rwidth=0.85)
    plt.title('Height Histogram')
    plt.xlabel('Height')
    plt.ylabel('Times')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(intervalos) 
    plt.show()

if __name__ == "__main__":
    main()