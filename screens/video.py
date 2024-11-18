import streamlit as st
import os
import time
from dotenv import load_dotenv
from utils.aux_functions import load_css, load_image, get_youtube_comments, preprocess_and_embed_bert
from transformers import BertTokenizer, BertModel
from datetime import datetime

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

load_dotenv()

def video_screen(stack_model_bert):
    load_css('style.css')

    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('logo 2.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class = "medium-font">Live prediction from youtube video.</p>', 
    unsafe_allow_html=True)

    video_url = st.text_input('Enter the youtube video URL here:')
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    col1, col2 = st.columns(2)

    with col1:
        get_comments_button = st.button('Get Comments')
    with col2:
        live_analysis_button = st.button('Live Analysis')

    if get_comments_button:
        comments = get_youtube_comments(video_url, api_key)
        hateful_comments = []
        for comment in comments:
            embedding = preprocess_and_embed_bert(comment, model, tokenizer)
            final_prediction, predictions = stack_model_bert.predict([embedding])
            if final_prediction == 1:
                hateful_comments.append(comment)
        
        if hateful_comments:
            st.write("Hateful comments detected:")
            for hateful_comment in hateful_comments:
                st.write(hateful_comment)
        else:
            st.write("No hateful comments detected.")

    if live_analysis_button:
        countdown_placeholder = st.empty()
        stop_button_placeholder = st.empty()
        comments_placeholder = st.empty()
        stop_button_key = 0
        seen_comments = {}  # Diccionario para almacenar comentarios ya vistos y su hora de extracción

        while True:
            stop = stop_button_placeholder.button('Stop', key=f'stop_button_{stop_button_key}')
            if stop:
                break
            else:
                comments = get_youtube_comments(video_url, api_key)
                new_hateful_comments = []
                for comment in comments:
                    if comment not in seen_comments:  # Solo procesar comentarios nuevos
                        embedding = preprocess_and_embed_bert(comment, model, tokenizer)
                        final_prediction, predictions = stack_model_bert.predict([embedding])
                        if final_prediction == 1:
                            new_hateful_comments.append(comment)
                            seen_comments[comment] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Agregar comentario al diccionario con la hora de extracción
                
                if new_hateful_comments:
                    for hateful_comment in new_hateful_comments:
                        comments_placeholder.write(f"{hateful_comment} (extracted at {seen_comments[hateful_comment]})")
                
                # Mostrar todos los comentarios vistos hasta ahora
                if seen_comments:
                    for comment, timestamp in seen_comments.items():
                        comments_placeholder.write(f"{comment} (extracted at {timestamp})")
                else:
                    comments_placeholder.write("No hateful comments detected.")
                
                for i in range(15, 0, -1):
                    countdown_placeholder.text(f"Updating comments in {i} seconds...")
                    time.sleep(1)
                
                stop_button_key += 1




    