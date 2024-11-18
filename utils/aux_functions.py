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
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import torch
from transformers import BertTokenizer, BertModel
from googleapiclient.discovery import build

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

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00a2bb"},
            'bgcolor': "#ffe6e6",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#3BE980'},
                {'range': [50, 75], 'color': '#E8ED47'},
                {'range': [75, 100], 'color': '#E34F24'}],
            'threshold': {
                'line': {'color': "#000e26", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(paper_bgcolor = "#ffe6e6", font = {'color': "#333333", 'family': "Arial"})    
    return fig

class MultiHeadHateClassifier_2:
    def __init__(self):
        self.models_random_forest = {}
        self.models_xgboost = {}
        self.label_columns = ['IsAbusive', 'IsProvocative', 'IsHatespeech', 'IsRacist']
        self.embeddings_index = None
        # Add SMOTE for handling imbalanced data
        self.smote = SMOTE(random_state=42)
        # Add cross-validation
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def optimize_random_forest_model(self, X, y, column):
        X_array = np.array(X)
        y_array = np.array(y[column])
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Increased range
                'max_depth': trial.suggest_int('max_depth', 5, 30),          # Increased range
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }
            
            scores = []
            for train_idx, val_idx in self.cv.split(X_array, y_array):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                # Apply SMOTE only on training data
                X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
                
                model = RandomForestClassifier(**params)
                model.fit(X_train_resampled, y_train_resampled)
                y_pred = model.predict(X_val)
                scores.append(f1_score(y_val, y_pred))
            
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)  # Increased trials
        
        # Get best parameters and train final model
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params)
        X_resampled, y_resampled = self.smote.fit_resample(X_array, y_array)
        best_model.fit(X_resampled, y_resampled)
        return best_model
    
    def optimize_xgboost_model(self, X, y, column):
        X_array = np.array(X)
        y_array = np.array(y[column])
        pos_weight = (y_array == 0).sum() / (y_array == 1).sum()

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': pos_weight,
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'tree_method': 'hist'  # For faster training
            }
            
            scores = []
            for train_idx, val_idx in self.cv.split(X_array, y_array):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
                
                model = XGBClassifier(**params,  early_stopping_rounds=20)
                model.fit(
                    X_train_resampled, 
                    y_train_resampled,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                y_pred = model.predict(X_val)
                scores.append(f1_score(y_val, y_pred))
            
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=75)  # Increased trials
        
        best_params = study.best_params
        best_model = XGBClassifier(**best_params)
        X_resampled, y_resampled = self.smote.fit_resample(X_array, y_array)
        best_model.fit(X_resampled, y_resampled)
        return best_model

    def fit(self, X, y):
        X = np.array(X)
        
        for column in self.label_columns:
            print(f"Optimizing Random Forest model for {column}")
            best_randomforest_model = self.optimize_random_forest_model(X, y, column)
            self.models_random_forest[column] = best_randomforest_model
            
            print(f"Optimizing XGBoost model for {column}")
            best_xgboost_model = self.optimize_xgboost_model(X, y, column)
            self.models_xgboost[column] = best_xgboost_model
            
    def predict(self, X):
        # Ensure X is a numpy array
        X = np.array(X)
        
        predictions = {column: {
            'rf_prob': self.models_random_forest[column].predict_proba(X)[:, 1],
            'xgb_prob': self.models_xgboost[column].predict_proba(X)[:, 1]
        } for column in self.label_columns}
        
        final_predictions = []
        for column in self.label_columns:
            rf_weight = 0.4
            xgb_weight = 0.6
            
            combined_preds = (
                (rf_weight * predictions[column]['rf_prob']) + 
                (xgb_weight * predictions[column]['xgb_prob'])
            )
            final_predictions.append(combined_preds > 0.5)
        
        final_predictions = np.array(final_predictions).T
        return np.any(final_predictions, axis=1)
    
    def load_embeddings(self, file_path):
        # Load GloVe embeddings
        self.embeddings_index = {}
        with open(file_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
                
    def preprocess_text(self, text):
        """Text preprocessing as defined in notebook"""
        tokens = procesar_texto(text) # Using existing function
        embedding = text_to_embedding(tokens, self.embeddings_index, 100)
        return np.array([embedding])
    

def preprocess_and_embed_bert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding 

class MultiHeadHateClassifier:
    def __init__(self):
        self.models = {}
        self.label_columns = ['IsAbusive', 'IsProvocative', 'IsHatespeech', 'IsRacist']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def optimize_model(self, X, y, column):
        """
        Realiza una búsqueda de hiperparámetros con Optuna para el modelo de una columna específica.
        """
        pos_weight = (y[column] == 0).sum() / (y[column] == 1).sum()

        def objective(trial):
            # Espacio de búsqueda para los hiperparámetros de XGBClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'scale_pos_weight': pos_weight
            }

            model = XGBClassifier(**params)
            model.fit(X, y[column])

            score = validation_score(model, X, y[column])
            return score

        # Crear el estudio Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25)

        # Guardar el mejor modelo en self.models
        best_params = study.best_params
        best_params['scale_pos_weight'] = pos_weight
        best_model = XGBClassifier(**best_params)
        best_model.fit(X, y[column])

        return best_model

    def fit(self, X, y):
        for column in self.label_columns:
            print(f"Optimizing model for {column}")
            model = self.optimize_model(X, y, column)
            self.models[column] = model

    def predict(self, X):
        predictions = {}
        for column in self.label_columns:
            predictions[column] = self.models[column].predict(X)

        final_prediction = np.any(list(predictions.values()), axis=0)
        return final_prediction, predictions

    def preprocess_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        return embedding
    
def get_youtube_comments(video_url, api_key):
    video_id = video_url.split('v=')[1]
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()
    
    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    
    return comments