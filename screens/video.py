import streamlit as st
import os
from dotenv import load_dotenv
from utils.aux_functions import load_css, load_image, get_youtube_comments, preprocess_and_embed_bert
from transformers import BertTokenizer, BertModel

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
    
    if st.button('Get Comments'):
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



    