import streamlit as st
from st_files_connection import FilesConnection

import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Converse with Data",
    page_icon="🦉",
    layout="wide"
)


TOGGLE = 's3' # 's3' OR 'local'
if TOGGLE=='s3':
    conn = st.experimental_connection('s3', type=FilesConnection)
else:
    conn = st.experimental_connection('local', type=FilesConnection)


st.markdown('''
# **Converse with Data**  

#### This is an **EDA App** created in Streamlit using the [PandasAI](https://github.com/gventuri/pandas-ai) library.
---
''')            

st.session_state.openai_key = st.secrets["openai_key"]
if "df" not in st.session_state:
    st.session_state.prompt_history = []
    st.session_state.df = None

if "openai_key" in st.session_state:
    st.sidebar.header("Choose dataset")
    st.sidebar.subheader("")
    filetype = st.radio("1- Select type of data",('CSV', 'Parquet'),1)

    if st.sidebar.button('Press to load an example dataset'):
        FILE_NAME = 'uci-cc-power-plant-data.csv'
        FILE_DESC = '''
                The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), 
                when the power plant was set to work with full load. \n
                Features consist of the following *hourly average ambient variables* to 
                predict the **net hourly electrical energy output (PE)** of the plant.
                - Temperature (T),  
                - Ambient Pressure (AP),  
                - Relative Humidity (RH)  and 
                - Exhaust Vacuum (V) to 
                [Visit UCI page](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
                '''
        if filetype=='CSV':
            if TOGGLE=='s3':
                df = conn.read(f"run-explorer-files/{FILE_NAME}", input_format="csv", ttl=600)
            else:
                df = conn.read((f'data/{FILE_NAME}'), input_format="csv", ttl=600)

        elif filetype=='Parquet': 
            FILE_NAME = 'items_w_agg_scores.parquet'
            FILE_DESC = f'''
            This data comes from **{TOGGLE.upper()}** parquet file
            '''
            if TOGGLE=='s3':
                df = conn.read(f"run-explorer-files/{FILE_NAME}", input_format="parquet", ttl=600)
            else:
                #df = pd.read_csv(f'data/{FILE_NAME}')
                df = conn.read((f'data/{FILE_NAME}'), input_format="parquet", ttl=600)

        st.session_state.df = df
        st.sidebar.markdown(FILE_DESC)
        
    st.sidebar.subheader('OR')
    uploaded_file = st.sidebar.file_uploader("",
        type="csv",
        help="Data should be in long format, one datapoint per row.",
        label_visibility='collapsed'
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

    if st.session_state.df is not None:
        with st.form("Question"):
            question = st.text_input("Question", value="", type="default")
            submitted = st.form_submit_button("Submit")
            st.markdown('''Examples of questions include:  
                        * Does the data have any missing values?  
                        * What is the average and standard deviation of each variable? Display the output in a dataframe.
                        * and make the text red for the largest value per column
                        ''')
            if submitted:
                with st.spinner():
                    llm = OpenAI(api_token=st.session_state.openai_key)
                    pandas_ai = PandasAI(llm) #, verbose=True, conversational=False, enforce_privacy=False, enable_cache=True)
                    x = pandas_ai.run(st.session_state.df, prompt=question)

                    fig = plt.gcf()
                    #fig, ax = plt.subplots()
                    #ax.hist(arr, bins=20)
                    if fig.get_axes():
                        st.pyplot(fig)
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

# SAMPLE QUESTION: List the average values for each variety and make the text red for the largest value per column