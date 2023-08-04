import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

st.markdown('''
# **Converse with Data**  

#### This is an **EDA App** created in Streamlit using the  [PandasAI](https://github.com/gventuri/pandas-ai) library.
---
''')            

st.session_state.openai_key = st.secrets["openai_key"]
st.session_state.prompt_history = []
if "df" not in st.session_state:
    st.session_state.df = None

if "openai_key" in st.session_state:
    if st.button('Press to use Example Dataset'):
        df = pd.read_csv('data/iris.csv')
        st.session_state.df = df

    uploaded_file = st.file_uploader(
        "Choose a CSV file. This should be in long format (one datapoint per row).",
        type="csv",
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

    if st.session_state.df is not None:
        with st.form("Question"):
            question = st.text_input("Question", value="", type="default")
            submitted = st.form_submit_button("Submit")
            if submitted:
                with st.spinner():
                    llm = OpenAI(api_token=st.session_state.openai_key)
                    pandas_ai = PandasAI(llm)
                    x = pandas_ai.run(st.session_state.df, prompt=question)

                    fig = plt.gcf()
                    if fig.get_axes():
                        st.pyplot(fig)
                        #st.write(fig)
                    st.write(x)
                    st.session_state.prompt_history.append(question)

    if st.session_state.df is not None:
        st.subheader("Current dataframe:")
        st.write(st.session_state.df)

    st.subheader("Prompt history:")
    st.write(st.session_state.prompt_history)


if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None