# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:35:36 2020

@author: Harish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
wine=pd.read_csv("wine.csv")
wine.describe()
wine_data= wine.iloc[:,1:]

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
wine_normal=scale(wine_data)

pca=PCA()
pca_values=pca.fit_transform(wine_normal)
pca_values.shape
var= pca.explained_variance_ratio_
var
pca.components_[0]
var_cum=np.cumsum(np.round(var,decimals=4)*100)
var_cum
#upto 7th column we get about 90% of data so we can discard remaining columns

plt.plot(var_cum,color="red")

x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
plt.plot(x,y,"ro")

#############kmeans-Clustering (original data)##############
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist

def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
df_norm=norm_func(wine.iloc[:,1:])


k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# elbow curve showing an elbow behaviour at cluster= 3 so the value of k=3
# optimum k value= 3
modelk= KMeans(n_clusters=3).fit(df_norm)
modelk.labels_
md=pd.Series(modelk.labels_) 
md.value_counts() 
wine['clust']=md
wine = wine.iloc[:,[0,14,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.groupby(wine.clust).mean()
wine_meank=wine.groupby(wine.clust).mean()



######################kmeans using 3 column #################
wine3=wine.iloc[:,[0,2,3,4,]]
def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
df_norm1=norm_func(wine3.iloc[:,1:])


k = list(range(2,15))
k
TWSS = []  
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm1)
    WSS = []  
    for j in range(i):
        WSS.append(sum(cdist(df_norm1.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm1.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# here also we are getting no of clusters= 3
# optimum k value=3


############## Hierarchical Clustering (original dataset)#################
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch


z = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()

from sklearn.cluster import	AgglomerativeClustering 
h_complete	= AgglomerativeClustering(n_clusters=3,linkage='single',affinity = "euclidean").fit(df_norm) 

h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
cluster_labels.value_counts()
wine['clust_h']=cluster_labels
wine = wine.iloc[:,[0,1,15,2,3,4,5,6,7,8,9,10,11,12,13,14]]

wine_meanh=wine.groupby(wine.clust).mean()

######################## Hierarchical Clustering using 3 column####################

z = linkage(df_norm1, method="single",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()

from sklearn.cluster import	AgglomerativeClustering 
h_complete1	= AgglomerativeClustering(n_clusters=3,linkage='single',affinity = "euclidean").fit(df_norm1) 

h_complete1.labels_
cluster_labels1=pd.Series(h_complete1.labels_)
cluster_labels1.value_counts()
# we get 3 clusters
# we are getting same no of clusters using Kmeans and hierarchical Clustering 
# by using original data set and dataset got from PCA