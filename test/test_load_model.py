import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.aux_functions import MultiHeadHateClassifier_2

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model('models/stack_model.pkl')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")