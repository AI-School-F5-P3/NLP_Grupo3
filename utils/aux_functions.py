import numpy as np
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
import os
from PIL import Image

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