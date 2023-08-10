import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


st.markdown('''
# **Converse with Data**  

#### This is an **EDA App** created in Streamlit using the  [PandasAI](https://github.com/gventuri/pandas-ai) library.
---
''')            

st.session_state.openai_key = st.secrets["openai_key"]
if "df" not in st.session_state:
    st.session_state.prompt_history = []
    st.session_state.df = None

if "openai_key" in st.session_state:
    if st.sidebar.button('Press to load an example dataset'):
        df = pd.read_csv('data/uci-cc-power-plant-data.csv')
        st.session_state.df = df
        st.sidebar.markdown('''
        The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), 
        when the power plant was set to work with full load. \n
        Features consist of the following *hourly average ambient variables* to 
        predict the **net hourly electrical energy output (PE)** of the plant.
        - Temperature (T),  
        - Ambient Pressure (AP),  
        - Relative Humidity (RH)  and 
        - Exhaust Vacuum (V) to 
        

        [Visit UCI page](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
                    
        ''')
    st.sidebar.subheader('OR')
    uploaded_file = st.sidebar.file_uploader("",
        type="csv",
        help="Data should be in long format, one datapoint per row.",
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

                    #fig = plt.gcf()
                    fig, ax = plt.subplots()
                    #ax.hist(arr, bins=20)
                    # if fig.get_axes():
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