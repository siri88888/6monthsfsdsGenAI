
import streamlit as st
import nltk
import os
# App title
st.title("NLU Stemmer ")
st.markdown(":red-background[ **Enter the text to stem **].")

# Two number input boxes
st.markdown(
    """
    <style>
    .stTextInput {
        background-color: yellow;
        color: black;
        border: 2px solid #4682b4; /* Steel blue border */
        border-radius: 5px; /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)

porter = st.text_input("Enter your string:")
#Lancster=st.text_input("Enter your string1:")
#Snowballstemmer=st.text_input("Enter your string:")



from nltk.stem  import PorterStemmer
pst=PorterStemmer()
from nltk.stem  import LancasterStemmer
lst=LancasterStemmer()
from nltk.stem  import SnowballStemmer
sbst=SnowballStemmer('english')

# Calculate button
st.markdown(
"""
<style>
div.stButton > button:first-child {
background-color: #00cc00;
color: white;
font-size: 20px;
height: 1em;
width: 10em;
border-radius: 10px 10px 10px 10px;
}
</style>
""",
unsafe_allow_html=True
)

if st.button("Stemmer"):
    st.write('Porter stemmer   '+porter+':' +pst.stem(porter))
    st.write('Lancaster stemmer  '+porter+':' +lst.stem(porter))
    st.write('Snowball stemmer   '+porter+':' +sbst.stem(porter))
    

    