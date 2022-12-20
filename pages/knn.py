import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st

data = sns.load_dataset("iris")

X = data.values[:,:4]
y = data.values[:,4]

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=4)
PC = pca.fit_transform(X)

pca.explained_variance_ratio_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train

fig = plt.figure()

plt.scatter(PC[:,0],PC[:,1])
st.pyplot(fig)
plt.close()


