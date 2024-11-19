import streamlit as st
from utils.aux_functions import load_css, load_image

def about_screen():
    load_css('style.css')
    
    image = load_image('logo 2.png')
    st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{image}" width="150">
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown('## About Us', unsafe_allow_html=True)