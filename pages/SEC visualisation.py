import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import streamlit as st

st.title("SEC visualisation")

path = '/Users/matthesfogtmann/Downloads/SEC data/'


def getDir(path):
    lst = []
    for i in os.listdir(path):
        if len(i)==6 and "q" in i:
            lst.append(i)
    return lst

def getFinacialData(filename=path+'2022q4'):
    data = {i:pd.read_csv(f"{filename}{i}.txt",sep="\t") for i in ["tag","num","pre","sub"]}
    
    #drop things
    data["tag"].drop(columns=["version","custom","abstract","datatype","iord","crdr"],inplace=True)
    data["num"].drop(columns=["footnote"],inplace=True)
    data["pre"].drop(columns=["version","negating"],inplace=True)
    data["sub"].drop(columns=['stprba', 'zipba',
       'bas1', 'bas2', 'baph', 'countryma', 'stprma', 'cityma', 'zipma',
       'mas1', 'mas2', 'ein'],inplace=True)
    
    #make mapper for company names
    
    def mapper(colname):
    	return {i:j for i,j in zip(data["sub"]["adsh"],data["sub"][colname])}
    
    def inverseMapper(colname):
        return {j:i for i,j in zip(data["sub"]["adsh"],data["sub"][colname])}
    
    data["num"]["name"] = data["num"]["adsh"].map(mapper("name"))
    
    data["num"].drop(columns=["version","coreg","adsh"],inplace=True)
    
    
    names = sorted(list(data["num"]["name"].unique()))[:10]

    def f(data, names):
        dic = dict()
        for name in tqdm(names):
            df = data["num"][data["num"]["name"]==name].copy()
            dic[name]= dict()
            
            # fill more data
            
            
            for i in set(df["tag"]):
                df_temp = df[df["tag"]==i]
                df_temp = df_temp[df_temp["ddate"]==max(df_temp["ddate"]) ]
                dic[name][list(df_temp["tag"])[0]]=df_temp["value"].values[0]
        return dic
    return f(data, names)
    return pd.DataFrame.from_dict(f(data, names)).T

def combineReports(reports):
    dic = {}
    for report_name, report in reports.items():
        for company in report.keys():
            if company not in dic:
                dic[company] = {report_name:report[company]}
            else:
                dic[company][report_name] = report[company]
                
    return pd.concat({k: pd.DataFrame(v).T for k,v in dic.items()},axis=0)

@np.vectorize
def timeString2float(x="2020q2"):
    lst = x.split("q")
    return int(lst[0])+(int(lst[1])-1)*0.25

df = pd.read_csv("data/SEC data.csv",index_col=[0,1])

options = [i for i in set(np.array(list(df.index))[:,0])]
companies = st.multiselect(label="Search for company",options=options)

if len(companies)>0:
    st.dataframe(df.loc[companies])

tags_options = [i for i in df.columns]
tags = st.multiselect(label="What tag",options=tags_options)


def plotCompanyTag(companies="1 800 FLOWERS COM INC",tags="Assets"):

    @np.vectorize
    def timeString2float(x="2020q2"):
        lst = x.split("q")
        return int(lst[0])+(int(lst[1])-1)*0.25

    try:
    
        if len(tags)>1:
            fig, ax = plt.subplots(len(tags),1,figsize=(10,8),dpi=300)

            for i in range(len(tags)):
                for company in companies:
                    y = df.loc[company][tags[i]].values
                    x = timeString2float(np.array(df.loc[company][tags[i]].index))

                    sort_index = np.argsort(x)

                    x = x[sort_index]
                    y = y[sort_index]

                    ax[i].plot(x,y,label=company)

                    ax[i].legend()
                    ax[i].set_ylabel(tags[i])
        else:
            fig, ax = plt.subplots(1,1,figsize=(16,6),dpi=300)
            for company in companies:
                y = df.loc[company][tags[0]].values
                x = timeString2float(np.array(df.loc[company][tags[0]].index))

                sort_index = np.argsort(x)

                x = x[sort_index]
                y = y[sort_index]

                ax.plot(x,y,label=company)

                ax.legend()
                ax.set_ylabel(tags[0])
        fig.tight_layout()
        st.pyplot(fig)

    except:
        pass

if len(tags)>0:
    plotCompanyTag(companies,tags)