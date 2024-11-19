import os
from pathlib import Path
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = BASE_DIR / 'akroma-id.json'
PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')

# Verificar si la aplicación de Firebase ya está inicializada
if not firebase_admin._apps:
    cred = credentials.Certificate(CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {
        'projectId': PROJECT_ID,
    })

db = firestore.client()

FIREBASE_SCHEMA = {
    "predictions": {
        "fields": {
            "text": "string",
            "model_name": "string",
            "vectorizer": "string",
            "prediction": "integer",
            "timestamp": "timestamp"
        }
    },
    "live_detection": {
        "fields": {
            "text": "string",
            "model_name": "string",
            "vectorizer": "string",
            "prediction": "integer",
            "timestamp": "timestamp"
        }
    } 
}

def initialize_firestore_schema():
    for collection_name, collection_schema in FIREBASE_SCHEMA.items():
        doc_ref = db.collection(collection_name).document("schema")
        doc_ref.set({
            "fields": collection_schema["fields"]
        })
        print(f"Initialized schema for collection: {collection_name}")

# Llamar a la función para inicializar el esquema
initialize_firestore_schema()

def save_prediction(text, model_name, vectorizer, prediction):
    doc_ref = db.collection("predictions").document()
    doc_ref.set({
        "text": text,
        "model_name": model_name,
        "vectorizer": vectorizer,
        "prediction": int(prediction),  
        "timestamp": firestore.SERVER_TIMESTAMP
    })

def save_live_detection(text, model_name, vectorizer, prediction):
    doc_ref = db.collection("live_detection").document()
    doc_ref.set({
        "text": text,
        "model_name": model_name,
        "vectorizer": vectorizer,
        "prediction": int(prediction),  
        "timestamp": firestore.SERVER_TIMESTAMP
    })