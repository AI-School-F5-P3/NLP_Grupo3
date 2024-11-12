import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import classification_report, accuracy_score
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.best_hyperparms = None
        self.vectorizer = None

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        return df
    
    def process_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def preprocess_data(self, df):
        df['processed_text'] = df['Text'].apply(self.process_text)

        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(df['processed_text'])

        y = df['IsToxic']

        return X, y
    
    def objective(self, trial, X, y, n_splits = 5):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Entrenar modelo
            model = XGBClassifier(**params, early_stopping_rounds = 50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluar
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))
            print(f'media de kfold {np.mean(scores)}')
        
        return np.mean(scores)
    
    def find_best_params(self, X, y, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            n_jobs=-1
        )
        
        self.best_params = study.best_params
        print(f"Best accuracy: {study.best_value:.4f}")
        print("Best hyperparameters:", self.best_params)
        return self.best_params

    def train_model(self, X, y, params=None):
        if params is None:
            params = self.best_params

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = XGBClassifier(**params, early_stopping_rounds = 50)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )
        
        # Evaluar modelo final
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict(self, text):
        """Predice si un nuevo texto contiene discurso de odio"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Preprocesar y vectorizar el texto
        processed_text = self.process_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # Predecir
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]
        
        return prediction, probability

if __name__ == "__main__":
    # Crear instancia del modelo
    model = XGBoostModel()
    
    # Cargar y preprocesar datos
    df = model.load_data('toxic.csv')
    X, y = model.preprocess_data(df)
    
    # Encontrar mejores hiperparámetros
    best_params = model.find_best_params(X, y, n_trials=50)
    
    # Entrenar modelo final
    model.train_model(X, y, best_params)
