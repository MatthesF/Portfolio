import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import seaborn as sns
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.title("Kmeans from scratch")

st.markdown(
    "Kmeans is an unsupervised machine learning algorithm that is used to classify data into a specified number of clusters. The algorithm begins by placing a specified number of random points, known as centroids, within the data set. It then assigns each data point to the closest centroid and calculates the mean of the points within each cluster. The algorithm then moves the centroids to the mean of the points within each cluster and reassigns the data points to the closest centroid. This process is repeated until the centroids no longer move or a maximum number of iterations is reached."
    )

st.markdown(
    "* Number of categories: this is the number of clusters that the algorithm will divide the data into. \n * Max iter: this is the maximum number of iterations that the algorithm will perform before stopping. If the centroids do not move or the mean of the points within each cluster does not change, the algorithm will stop before reaching the maximum number of iterations."
)

def check_if_centroid_within_span(X,proposed_centroids):
    
    for c in proposed_centroids:
        if (np.min(X,axis=0)<c).all() and (np.max(X,axis=0)>c).all():
            pass
        else:
            return "BAD"
    return "GOOD"

def getAffillation(X,C,iterate_over = "C"):
    
    lengths = []
    
    if iterate_over == "C":
        for c in C:
            lengths.append(np.sum((X-c)**2,axis=1))
            
        return np.argmin(np.array(lengths), axis=0)
    
    else:
        for x in X:
            lengths.append(np.sum((C-x)**2,axis=1))
        
        return np.argmin(np.array(lengths), axis=1)


def updateCentroids(X,affiliation,num_of_c):

    return np.array([np.mean(X[affiliation==i],axis=0) for i in set(affiliation)])

def KMeans(X,num_of_c = 5,maxFev = 100,animate=True):
    
    dim = np.shape(X)[1]
    
    new_centroids_check = "BAD"

    while new_centroids_check == "BAD":
        proposed_centroids = np.random.randn(num_of_c, dim)*1+np.mean(X,axis=0)
        new_centroids_check = check_if_centroid_within_span(X,proposed_centroids)

    C = proposed_centroids

    if animate:
        fig = plt.figure()
        camera = Camera(fig)
        plt.scatter(X[:,0],X[:,1])
        plt.scatter(C[:,0],C[:,1],s=200,c="r")
        camera.snap()

    for i in range(maxFev):

        affiliation = getAffillation(X,C)

        updated_centroids = updateCentroids(X,affiliation,num_of_c)
        
        C = updated_centroids
        
        if animate:
            
            plt.scatter(X[:,0],X[:,1],c=affiliation)
            plt.scatter(C[:,0],C[:,1],s=200,c="red")
            camera.snap()


        if i > 0 and (affiliation == last_affiliation).all():
            
            if animate:
                for i in range(7):
                    plt.scatter(X[:,0],X[:,1],c=affiliation)
                    plt.scatter(C[:,0],C[:,1],s=200,c="red")
                    camera.snap()
                animation = camera.animate()
                components.html(animation.to_jshtml(), height=1000)
            return affiliation
        last_affiliation = affiliation

    return affiliation

df = sns.load_dataset("iris")
X_iris = df.values[:,:-1]

cols = st.columns(2)
num_of_c = cols[0].slider("Number of catagories",1,10,3)
maxFev = cols[1].slider("Max iter",0,500,100)

mapper = dict(zip(set(df["species"]),range(len(set(df["species"])))))
df["label"] = df.species.apply(lambda x: mapper[x])

affiliation = KMeans(X_iris,maxFev=maxFev,num_of_c=num_of_c,animate=True)
