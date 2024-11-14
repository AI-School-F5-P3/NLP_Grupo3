import streamlit as st
import pandas as pd

st.set_page_config(
    page_title= "Akroma",
    page_icon = "🕊️",
    layout = 'wide'
)

from pages.home import home_screen
from pages.predict import predict_screen
from pages.about import about_screen

if 'screen' not in st.session_state:
    st.session_state.screen = 'Home'

def change_screen(new_screen):
    st.session_state.screen = new_screen

st.sidebar.header('Navigation')
if st.sidebar.button('Home'):
    change_screen('Home')
if st.sidebar.button('Predict'):
    change_screen('Predict')
if st.sidebar.button('About'):
    change_screen('About')