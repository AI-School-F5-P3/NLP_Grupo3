import streamlit as st
from utils.aux_functions import load_css, load_image

def video_screen():
    load_css('style.css')

    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('logo 2.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class = "medium-font">Live prediction from youtube video.</p>', unsafe_allow_html=True)

    