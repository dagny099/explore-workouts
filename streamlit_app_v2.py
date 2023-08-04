import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np

def load_csv(flag=None):
    if flag:
        info = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'] )
    else:
        info = pd.read_csv(uploaded_file)
    return info

# Web App Title
st.markdown('''
# **The EDA App**

This is the **EDA App** created in Streamlit using the **Pandas AI** library.

---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", 
                                             type=["csv"])
    # if st.sidebar.button('Press to use Example Dataset'):
    #     # Example data
    #     df = load_csv('sample')

# Pandas Profiling Report
if uploaded_file is not None:
    df = load_csv()
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('** Next ...**')

else:
    st.info('Awaiting for CSV file to be uploaded.')

