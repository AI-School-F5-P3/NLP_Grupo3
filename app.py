import streamlit as st
from screens.home import home_screen
from screens.predict import predict_screen
from screens.about import about_screen
from utils.aux_functions import MultiHeadHateClassifier_2, load_model

if 'stack_model' not in st.session_state:
    try:
        st.session_state.stack_model = load_model('models/stack_model.pkl')
        print("Modelo cargado exitosamente en app.py")
    except Exception as e:
        print(f"Error al cargar el modelo en app.py: {e}")

st.set_page_config(
    page_title= "Akroma",
    page_icon = "🕊️",
    layout = 'wide'
)

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

if st.session_state.screen == 'Home':
    home_screen()
if st.session_state.screen == 'Predict':
    predict_screen(st.session_state.stack_model)
if st.session_state.screen == 'About':
    about_screen()