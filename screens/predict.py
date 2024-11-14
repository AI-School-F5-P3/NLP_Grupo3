import streamlit as st
from utils.aux_functions import load_css, load_image, load_glove_embeddings, preprocess_and_embed, load_model, create_gauge_chart
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors

xgb_model = load_model('models/xgb_model.pkl')

model_path = 'models/stack_model.pkl'
#stack_model = load_model(model_path)

embeddings_index = load_glove_embeddings('assets/glove.twitter.27B.100d.txt')

def predict_screen():
    load_css('style.css')

    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('logo 2.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h1 class="big-font">Hateful comments prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Write the comment to evaluate:</p>', unsafe_allow_html=True)

    text = st.text_input('Enter your text here:')

    selectbox = st.selectbox('Select a model', ('Simple XGBoost', 'MultiHead Stack'))

    if st.button('Make prediction'):
        if selectbox == 'Simple XGBoost':
            processed_text = preprocess_and_embed(text, embeddings_index, embedding_dim=100)

            prediction = xgb_model.predict(processed_text)[0]
            proba = xgb_model.predict_proba(processed_text)[0][1]
            
            if prediction == 0:
                st.success('Congratulations! This comment is not hateful.')
            else:
                st.error('Warning! This comment is hateful.')

            fig = create_gauge_chart(proba * 100, "Hatefulness probability")
            st.plotly_chart(fig, use_container_width=True)

        if selectbox == 'MultiHead Stack':
            processed_text = preprocess_and_embed(text, embeddings_index, embedding_dim=100)

            prediction = stack_model.predict(processed_text)[0]
            if prediction == 0:
                st.succes('Congratulations! This comment is not hateful.')
            else:
                st.error('Warning! This comment is hateful.')
