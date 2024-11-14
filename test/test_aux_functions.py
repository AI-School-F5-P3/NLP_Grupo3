import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from utils.aux_functions import load_glove_embeddings, procesar_texto, text_to_embedding, get_project_root, load_image, load_css, preprocess_and_embed, load_model, create_gauge_chart
from PIL import Image
import plotly.graph_objects as go

def test_load_glove_embeddings():
    embeddings = load_glove_embeddings('../assets/glove.twitter.27B.100d.txt')
    assert isinstance(embeddings, dict)
    assert 'the' in embeddings

def test_procesar_texto():
    text = "This is a simple test text, not too complicated!"
    tokens = procesar_texto(text)
    assert isinstance(tokens, list)
    assert 'this' not in tokens
    assert 'simpl' in tokens

def test_text_to_embedding():
    embeddings_index = {'test': np.array([1, 2, 3]), 'text': np.array([4, 5, 6])}
    tokens = ['test', 'text']
    embedding = text_to_embedding(tokens, embeddings_index, embedding_dim=3)
    np.testing.assert_array_equal(embedding, np.array([2.5, 3.5, 4.5]))

def test_get_project_root():
    root = get_project_root()
    assert root.endswith('NLP_Grupo3')

def test_load_image():
    image = load_image('../assets/logo.png')
    assert isinstance(image, Image.Image)

def test_load_css():
    try:
        load_css('../styles/style.css')
    except Exception as e:
        pytest.fail(f"load_css raised {e} unexpectedly!")

def test_preprocess_and_embed():
    embeddings_index = {'test': np.array([1, 2, 3]), 'text': np.array([4, 5, 6])}
    text = "This is a test text"
    embedding = preprocess_and_embed(text, embeddings_index, embedding_dim=3)
    assert embedding.shape == (1, 3)

def test_load_model():
    model = load_model('../models/xgb_model.pkl')
    assert model is not None

def test_create_gauge_chart():
    fig = create_gauge_chart(75, "Test Gauge")
    assert isinstance(fig, go.Figure)