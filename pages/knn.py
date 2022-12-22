
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st

plt.style.use('dark_background')

st.title("KNN from scratch")

st.markdown(
    "K-nearest neighbors (KNN) is a classification algorithm that works by identifying the k number of observations in the training data that are closest to a given test observation, and then predicting the class of the test observation based on the majority class among those neighbors."
    )

st.markdown("* Number of neighbors (k): The value of k determines the number of neighbors that will be used to make the prediction for a given test observation.\n * Test data size: The size of the test data set determines the number of observations that will be used to evaluate the performance of the model.")

def getDist2Neighbor(X_test,X_train):
    return np.sum(np.abs(X_train-X_test),axis=1)

def getNeighborsSpecies(X_test,X_train,n_neighbors):
    dist = getDist2Neighbor(X_test,X_train)
    index_dist = np.argsort(dist)[:n_neighbors]
    sorted_dist_index = [y_train[index] for index in index_dist]
    sorted_dist = np.sort(dist)[:n_neighbors]
    return (sorted_dist_index,sorted_dist)

def findMostCommonSpeceis(neighbor_species_list,dist):
    biggest = 0
    num = None
    for i in list(set(neighbor_species_list)):
        pos = np.where(neighbor_species_list==i)
        if (np.sum(np.max(y)-y[list(pos)])>biggest):
            biggest = np.sum(np.max(dist)-dist[list(pos)])
            num = i
    return num

def KNN(X_test,X_train,n_neighbors):
    neighbors = [getNeighborsSpecies(x,X_train,n_neighbors) for x in X_test]
    species = [findMostCommonSpeceis(neighbor,dist) for neighbor,dist in neighbors]
    return species

def plotDataKNN(X_test,X_train,y_test,n_neighbors,X_test_pca):
    species = KNN(X_test,X_train,n_neighbors)
    correct = 0
    
    correct_list = list()
    incorrect_list = list()

    for data,yHat,y in zip(X_test_pca,species,y_test):
        if yHat==y:
            correct+=1
            correct_list.append((list(data),yHat))
        else:
            incorrect_list.append((list(data),yHat))

    return correct_list,incorrect_list,(correct/len(y_test))

df = sns.load_dataset("iris")

# map species string to interger
mapper = dict(zip(set(df["species"]),range(len(set(df["species"])))))
df["label"] = df.species.apply(lambda x: mapper[x])
df.drop("species",axis=1,inplace=True)

# assign data and label
X = df.values[:,:-1]
y = df.values[:,-1]

# scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

cols = st.columns(2)
n_neighbors = cols[0].slider("Number of neighbors",1,100,2)
test_size = cols[1].slider("Data test size",0.02,.99,0.33)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# PCA
pca = PCA(n_components=2)
pca = pca.fit(X)

# transform data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

correct_list, incorrect_list, acc = plotDataKNN(X_test,X_train,y_test,n_neighbors,X_test_pca)

fig,ax = plt.subplots(figsize=(10,6),dpi=300)

# train data scatter
ax.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train,s=100,edgecolors='grey')

# correct labeled data scatter
correct_data = np.array([i[0] for i in correct_list])
ax.scatter(correct_data[:,0],correct_data[:,1],
           marker="X",c=[i[1] for i in correct_list],edgecolors='green', linewidth=2.5,s=150)

# incorrect labeled data scatter
incorrect_data = np.array([i[0] for i in incorrect_list])
ax.scatter(incorrect_data[:,0],incorrect_data[:,1],
           marker="X",c=[i[1] for i in incorrect_list],edgecolors='red', linewidth=2.5,s=150)


ax.set_title("PCA of iris")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])

cols[0].markdown(f"Accurarcy: {round(acc,3)}")

st.pyplot(fig)
plt.close()


