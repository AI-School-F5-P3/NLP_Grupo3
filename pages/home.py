import streamlit as st
from utils.aux_functions import load_css, load_image

def home_screen():
    load_css('style.css')

    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('/assets/logo.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class = "medium-font">Welcome to Akroma, your automated guardian angel.</p>', unsafe_allow_html=True)

    # Servicios
    st.markdown('## Our Services', unsage_allow_html=True)