from sklearn.decomposition import PCA
from pandas import read_csv, DataFrame
from sys import argv
from matplotlib.pyplot import scatter,show

# 2D PCA from csv

file = argv[1]
if len(argv) == 0 :
    print("no csv found \n usage 'python pca.py file.csv'")

#read file
data = read_csv(file,header = 0 , index_col = 0 )

#PCA
result = PCA(2)
result = result.fit_transform(data)

#export pca csv
df = DataFrame(result)
df.to_csv(argv[1][:-4]+"pca.csv")

#transpose
result = result.T

#plot
scatter(result[0],result[1])
show()