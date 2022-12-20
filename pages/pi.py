import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Hypersphere in hypercube")
cols = st.columns(2)
dim = cols[0].slider("Num dimensions",2,5,2)
size = cols[0].slider("Num datapoints",0,10000,250)
p = cols[0].slider("p",-1.,10.,2.)
data = np.random.uniform(-1,1,(size,dim))

dist = np.sum(np.abs(data)**p,axis=1)**(1/p)
cols[0].markdown(r"""
$$
d_p =\left(\sum_i x_i^p\right)^{1/p}
$$
""")

def within(dist):
    dist_list = []
    for i in dist:
        if i > 1:
            dist_list.append(1)
        else:
            dist_list.append(0)
    return dist_list     

def percentInside(dist_list):
    return (1-sum(dist_list)/len(dist_list))

dist_list = within(dist)

cols[0].write(f"Percent within unit circle {percentInside(dist_list)}")
fig = plt.figure()

plt.scatter(data[:,0],data[:,1],c=dist_list)

plt.close()

cols[1].pyplot(fig)

percent_list = []
for i in range(2,20):
    data = np.random.uniform(-1,1,(size,i))
    dist = np.sum(np.abs(data)**p,axis=1)**(1/p)
    dist_list = within(dist)
    percent_list.append(percentInside(dist_list))

fig = plt.figure()
plt.plot(range(2,20),percent_list)
cols[1].pyplot(fig)
plt.close()