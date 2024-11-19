import streamlit as st
import os
import time
from dotenv import load_dotenv
from utils.aux_functions import load_css, load_image, get_youtube_comments, preprocess_and_embed_bert, preprocess_and_embed, load_glove_embeddings, test_texts
from transformers import BertTokenizer, BertModel
from datetime import datetime
from database.firebase_config import save_prediction, save_live_detection

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

embeddings_index = load_glove_embeddings('assets/glove.twitter.27B.100d.txt')

load_dotenv()

def video_screen(xgb_model, stack_model, xgb_model_bert, stack_model_bert, bilstm_model):
    load_css('style.css')

    image = load_image('logo 2.png')
    st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{image}" width="150">
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown('<p class = "medium-font">Live prediction from youtube video.</p>', 
    unsafe_allow_html=True)

    video_url = st.text_input('Enter the youtube video URL here:')
    if not video_url.startswith("https://www.youtube.com/") and not video_url.startswith("https://youtu.be/"):
        st.error("Please enter a valid YouTube URL.")
        return
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    col1, col2 = st.columns(2)

    with col1:
        selectbox = st.selectbox('Select a model', ('Simple XGBoost', 'MultiHead Stack', 'Bidirectional LSMT'))
    with col2:
        if (selectbox == 'Simple XGBoost') | (selectbox == 'MultiHead Stack'):
            selectbox_2 = st.selectbox('Select embedding model', ('GloVe', 'BERT'))
        else:
            selectbox_2 = 'BERT'

    col3, col4 = st.columns(2)

    with col3:
        get_comments_button = st.button('Get Comments')
    with col4:
        live_analysis_button = st.button('Live Analysis')

    if get_comments_button:
        comments = get_youtube_comments(video_url, api_key)
        hateful_comments = []
        for comment in comments:
            if (selectbox == 'Simple XGBoost') & (selectbox_2 == 'GloVe'):
                processed_text = preprocess_and_embed(comment, embeddings_index, embedding_dim=100)
                prediction = xgb_model.predict(processed_text)[0]
                save_prediction(comment, selectbox, selectbox_2, prediction)
                if prediction == 1:
                    hateful_comments.append(comment)

            elif (selectbox == 'MultiHead Stack') & (selectbox_2 == 'GloVe'):
                processed_text = preprocess_and_embed(comment, embeddings_index, embedding_dim=100)
                prediction = stack_model.predict(processed_text)[0]
                save_prediction(comment, selectbox, selectbox_2, prediction)
                if prediction == 1:
                    hateful_comments.append(comment)

            elif (selectbox == 'Simple XGBoost') & (selectbox_2 == 'BERT'):
                embedding = preprocess_and_embed_bert(comment, model, tokenizer)
                final_prediction = xgb_model_bert.predict([embedding])[0]
                save_prediction(comment, selectbox, selectbox_2, final_prediction)
                if final_prediction == 1:
                    hateful_comments.append(comment)

            elif (selectbox == 'MultiHead Stack') & (selectbox_2 == 'BERT'):
                embedding = preprocess_and_embed_bert(comment, model, tokenizer)
                final_prediction, predictions = stack_model_bert.predict([embedding])
                save_prediction(comment, selectbox, selectbox_2, final_prediction)
                if final_prediction == 1:
                    hateful_comments.append(comment)

            elif (selectbox == 'Bidirectional LSMT'):
                final_prediction = test_texts(comment, bilstm_model, tokenizer, model)
                save_prediction(comment, selectbox, selectbox_2, final_prediction)
                if final_prediction > 0.5:
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
                        if (selectbox == 'Simple XGBoost') & (selectbox_2 == 'GloVe'):
                            processed_text = preprocess_and_embed(comment, embeddings_index, embedding_dim=100)
                            prediction = xgb_model.predict(processed_text)[0]
                            if prediction == 1:
                                new_hateful_comments.append(comment)
                                seen_comments[comment] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_live_detection(comment, selectbox, selectbox_2, prediction)


                        elif (selectbox == 'MultiHead Stack') & (selectbox_2 == 'GloVe'):
                            processed_text = preprocess_and_embed(comment, embeddings_index, embedding_dim=100)
                            prediction = stack_model.predict(processed_text)[0]
                            if prediction == 1:
                                new_hateful_comments.append(comment)
                                seen_comments[comment] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_live_detection(comment, selectbox, selectbox_2, prediction)

                        elif (selectbox == 'Simple XGBoost') & (selectbox_2 == 'BERT'):
                            embedding = preprocess_and_embed_bert(comment, model, tokenizer)
                            final_prediction = xgb_model_bert.predict([embedding])[0]
                            if final_prediction == 1:
                                new_hateful_comments.append(comment)
                                seen_comments[comment] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_live_detection(comment, selectbox, selectbox_2, final_prediction)

                        elif (selectbox == 'MultiHead Stack') & (selectbox_2 == 'BERT'):
                            embedding = preprocess_and_embed_bert(comment, model, tokenizer)
                            final_prediction, predictions = stack_model_bert.predict([embedding])
                            if final_prediction == 1:
                                new_hateful_comments.append(comment)
                                seen_comments[comment] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_live_detection(comment, selectbox, selectbox_2, final_prediction)

                        elif (selectbox == 'Bidirectional LSMT'):
                            final_prediction = test_texts(comment, bilstm_model, tokenizer, model)
                            if final_prediction > 0.5:
                                new_hateful_comments.append(comment)
                                seen_comments[comment] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_live_detection(comment, selectbox, selectbox_2, final_prediction)
                
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




    