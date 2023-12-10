from pandas import read_csv
from sys import argv
from matplotlib.pyplot import scatter,show
from random import randint
from numpy import vstack , array

#Dotplot from csv

file = argv[1]
if len(argv) == 0 :
    print("no csv found \n usage 'python dotplot.py file.csv'")

#read file
data = read_csv(file,header = 0 , index_col = 0 )
data = data.to_numpy()
data = data.T

# add class
data = vstack([data,[randint(0,4) for i in range(len(data[0]))]])
print(data)
print(len(data))

#plot
if len(data) == 3:
    scatter(data[0],data[1],c = data[2])
elif len(data) == 2 :
    scatter(data[0],data[1])
else :
    print("error , wrong data length")
show()