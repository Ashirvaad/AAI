import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
customers_data=pd.read_csv("C:/Users/tcsc/Desktop/Mall_Customers.csv")
customers_data.shape
customers_data.head()
data=customers_data.iloc[:,3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("Customer Dendograms")
dend=shc.dendogram(shc.linkage(data,method='ward'))
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=5,linkage='ward')
cluster.fit_predict(data)
plt.figure(figsize=(10,7))
plt.scatter(data[:,0],data[:,1],c=cluster.labels_,cmap='rainbow')
plt.show()