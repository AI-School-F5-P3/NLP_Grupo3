import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from utils.aux_functions import load_glove_embeddings, procesar_texto, text_to_embedding, get_project_root, load_image, load_css, preprocess_and_embed, load_model, create_gauge_chart, get_youtube_comments, preprocess_and_embed_bert, MultiHeadHateClassifier_2, MultiHeadHateClassifier, BiLSTMModel, text_to_embedding_nn, procesar_texto_nn
from PIL import Image
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

def test_load_glove_embeddings():
    embeddings = load_glove_embeddings('assets/glove.twitter.27B.100d.txt')
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
    image = load_image('logo.png')
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
    model = load_model('models/xgb_model.pkl')
    assert model is not None

def test_create_gauge_chart():
    fig = create_gauge_chart(75, "Test Gauge")
    assert isinstance(fig, go.Figure)

def test_get_youtube_comments(mocker):
    mocker.patch('utils.aux_functions.build')
    mock_youtube = mocker.Mock()
    mocker.patch('utils.aux_functions.build', return_value=mock_youtube)
    mock_request = mocker.Mock()
    mock_youtube.commentThreads().list.return_value = mock_request
    mock_request.execute.return_value = {
        'items': [
            {'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Test comment 1'}}}},
            {'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Test comment 2'}}}}
        ]
    }
    
    comments = get_youtube_comments('https://www.youtube.com/watch?v=test_video_id', 'fake_api_key')
    assert comments == ['Test comment 1', 'Test comment 2']

def test_preprocess_and_embed_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    text = "This is a test text"
    embedding = preprocess_and_embed_bert(text, model, tokenizer)
    assert embedding.shape == (768,)

def test_multiheadhateclassifier_2():
    classifier = MultiHeadHateClassifier_2()
    assert classifier is not None

def test_multiheadhateclassifier():
    classifier = MultiHeadHateClassifier()
    assert classifier is not None

def test_bilstm_model():
    model = BiLSTMModel(input_dim=100, hidden_dim=50, output_dim=4, num_layers=2, dropout_rate=0.5)
    assert isinstance(model, nn.Module)

def test_text_to_embedding_nn():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    text = "This is a test text"
    embedding = text_to_embedding_nn(text, model, tokenizer)
    assert embedding.shape == (768,)

def test_procesar_texto_nn():
    text = "This is a simple test text, not too complicated!"
    processed_text = procesar_texto_nn(text)
    assert isinstance(processed_text, str)
    assert 'this' not in processed_text
    assert 'simpl' in processed_text

def test_load_image():
    image = load_image('logo.png')
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
    model = load_model('models/xgb_model.pkl')
    assert model is not None

def test_create_gauge_chart():
    fig = create_gauge_chart(75, "Test Gauge")
    assert isinstance(fig, go.Figure)

def test_get_youtube_comments(mocker):
    mocker.patch('utils.aux_functions.build')
    mock_youtube = mocker.Mock()
    mocker.patch('utils.aux_functions.build', return_value=mock_youtube)
    mock_request = mocker.Mock()
    mock_youtube.commentThreads().list.return_value = mock_request
    mock_request.execute.return_value = {
        'items': [
            {'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Test comment 1'}}}},
            {'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Test comment 2'}}}}
        ]
    }
    
    comments = get_youtube_comments('https://www.youtube.com/watch?v=test_video_id', 'fake_api_key')
    assert comments == ['Test comment 1', 'Test comment 2']

def test_preprocess_and_embed_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    text = "This is a test text"
    embedding = preprocess_and_embed_bert(text, model, tokenizer)
    assert embedding.shape == (768,)