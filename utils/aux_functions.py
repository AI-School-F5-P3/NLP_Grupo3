import numpy as np
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
import os
import pickle
from PIL import Image
import plotly.graph_objects as go

def load_glove_embeddings(file_path):
    embedding_index = {}
    with open(file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index

def procesar_texto(text):
    '''
    1. El texto se pasa todo a minúsculas
    2. Se eliminan todos los símbolos que pueden dar problemas
    3. Se tokeniza el texto
    4. Se eliminan las stop_words, básicamente preposiciones o palabras que no aportan contenido semántico, a excepción de las negaciones porque pueden dar información conrextual relevante.
    5. Se convierten los tokens mediantes stemming, que se queda con la raiz o "lema" de una palabra para mejorar el rendimiento
    '''
    text = text.lower() 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def text_to_embedding(tokens, embeddings_index, embedding_dim=100):
    '''
    1. Se inicializa el embedding (vector) vacío pero con las dimensiones especificadas (100, que coincide con las dimensiones de los embeddings descargados)
    Este vector guarda el sumatorio acumulado de cada embedding individual de cada palabra (de 100 dimensiones también), que representará el texto completo.
    2. Se inicia un conteo a 0, este conteo servirá para contar cuantos embeddings válidos encontramos en el índice, y si no hay se mantiene a 0 para prevenir errores futuros al dividir.
    3. Se itera sobre la lista de tokens del texto y si se encuentra el token en el índice de embeddings, se añade el vector de la palabra al vector de embedding. Cada vez que encuentra una válido se suma 1 al conteo.
    4. Por último se saca la media del embedding con el número de conteo para que las palabras más representadas en un texto no dominen el embedding final.
    '''
    embedding = np.zeros(embedding_dim)
    count = 0
    for token in tokens:
        if token in embeddings_index:
            embedding += embeddings_index[token]
            count += 1
    if count > 0:
        embedding /= count
    return embedding

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def load_image(image_name):
    project_root = get_project_root()
    image_path = os.path.join(project_root, 'assets', image_name)
    return Image.open(image_path)

def load_css(file_name):
    project_root = get_project_root()
    css_path = os.path.join(project_root, 'styles', file_name)
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def preprocess_and_embed(text, embeddings_index, embedding_dim=100):
    # Preprocesa el texto
    tokens = procesar_texto(text)
    # Convierte el texto en los embeddings como vimos en el princpio
    embedding = text_to_embedding(tokens, embeddings_index, embedding_dim)
    return np.array([embedding])  # Devuelve un array 2d para las predicciones

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        steps = [
            {'range': [0, 50], 'color': '#3BE980'},
            {'range': [50, 75], 'color': '#E8ED47'},
            {'range': [75, 100], 'color': '#E34F24'}],
        threshold = {
            'line': {'color': "#000e26", 'width': 4},
            'thickness': 0.75,
            'value': 90},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00a2bb"},
            'bgcolor': "#000e26",
            'borderwidth': 2,
            'bordercolor': "white"}))
    
    fig.update_layout(paper_bgcolor = "#000e26", font = {'color': "white", 'family': "Arial"})
    
    return fig