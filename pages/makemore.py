import streamlit as st
import numpy as np
import string
import matplotlib.pyplot as plt


def makemore():
    st.title("Makemore")
    st.markdown("Welcome to Makemore, a Python project that aims to generate new names using a variety of techniques. The goal of this project is to explore different methods of generating names, from simple algorithms to more complex techniques.")

def bigram():
    st.title("Bigram")
    cols = st.columns(2)
    cols[0].markdown("The Bigram algorithm works by analyzing the frequency of pairs of characters in the training data, and using this information to make predictions about the next character in a sequence. While this approach can be effective in generating realistic and creative names, it does have a limitation in that it does not look back at previous characters in the sequence. This means that the algorithm may not always generate names that are coherent, as it does not consider the context in which each character appears.")



    def find_letter_combs(file_dir='data/names.txt',plot=False):
    
        with open(file_dir) as f:
            text = f.read().lower().splitlines()
        
        names = ["."+i+"." for i in text]
        
        lc = {}
        for n in names:
            for i,j in zip(n,n[1:]):
                c = "".join([i,j])
                if c not in lc:
                    lc[c] = 1
                else:
                    lc[c] += 1
                    
        letters = "."+string.ascii_lowercase
        letter_index = dict(zip(letters,range(len(letters))))
        letter_list = [i for i in letters]
        
        
        ocr_arr = np.zeros((len(letters),len(letters)))
        for i in lc:
            ocr_arr[letter_index[i[0]]][letter_index[i[1]]] = lc[i]
        
        def heatmap():
            fig, ax = plt.subplots(1,1,figsize=(5,5),dpi=150)
            
            pos = ax.imshow(ocr_arr,cmap="plasma")
            fig.colorbar(pos, ax=ax)
            ax.set(xticks=range(len(letters)),xticklabels = letters,xlabel = "Second letter",
                yticks=range(len(letters)),yticklabels = letters,ylabel = "First letter")
            
            cols[1].pyplot(fig)
        if plot: heatmap()

        return ocr_arr,letter_index,letter_list

    def make_words(ocr_arr,letter_index,letter_list):
        word = "."
        while word.count(".")<2:
            follow_prop = ocr_arr[letter_index[word[-1]]] / np.sum(ocr_arr[letter_index[word[-1]]])
            word = "".join([word,np.random.choice(letter_list,p=follow_prop)])
            
        if len(word)<5 or len(word)>12:
            word = make_words(ocr_arr,letter_index,letter_list)
        return word.strip(".")
    
    ocr_arr,letter_index,letter_list = find_letter_combs(plot=True)

    def makeNames(num):
        names_list = []
        for i in range(num):
            names_list.append(make_words(ocr_arr,letter_index,letter_list).capitalize())
        return ",  ".join(names_list)

    cols[1].markdown("Shows frequency of letter combinations")
    cols[1].write(". indicates start or end of combination")

    num = st.slider("Number of names",1,25,5)

    st.write(makeNames(num))
    



method_dict = {
    'makemore': makemore,
    'bigram': bigram,
    }

method = st.sidebar.selectbox("method" , list(method_dict.keys()))

run_method = method_dict[method]

run_method()