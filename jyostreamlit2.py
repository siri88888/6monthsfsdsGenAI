

import streamlit  as  st
import pandas as pd
import numpy as np
st.title("     my first  streamlit app")
st.write("Welcome  this app basic functionalities of streamlit")
st.sidebar.header("User Input features")
user_name =st.sidebar.text_input("what is your name","Jyothi here")
age=st.sidebar.slider("selectb your age",0,100,25)
favcolor=st.sidebar.selectbox("what is your fav color",["blue","red","green"])
st.header(f"welcome,{user_name}.")
st.write(f"your {age} is older than   your fav color is  {favcolor}.")
st.subheader("here is some data")

data=pd.DataFrame(
       np.random.randn(10,5),
      columns=('col %d' % i for i  in range(5))
)
st.dataframe(data)
if st.checkbox("show raw data"):
    st.subheader("raw data")
    st.write(data)