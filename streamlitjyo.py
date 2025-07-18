
import streamlit  as  st
st.title("my first  streamlit app")
st.write("Welcome  this app calculates the square of  a number")
st.header("Select a number")
num=st.slider("pick a number",0,100,25)
st.subheader("Result")
sqnum=num*num
st.write(f"The square of  **{sqnum}**. ")