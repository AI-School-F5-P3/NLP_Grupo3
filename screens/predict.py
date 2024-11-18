import streamlit as st
from utils.aux_functions import load_css, load_image, load_glove_embeddings, preprocess_and_embed, create_gauge_chart, preprocess_and_embed_bert
from transformers import BertTokenizer, BertModel


embeddings_index = load_glove_embeddings('assets/glove.twitter.27B.100d.txt')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def predict_screen(xgb_model, stack_model, xgb_model_bert, stack_model_bert):
    load_css('style.css')

    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('logo 2.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h1 class="big-font">Hateful comments prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Write the comment to evaluate:</p>', unsafe_allow_html=True)

    text = st.text_input('Enter your text here:')

    col1, col2 = st.columns(2)

    with col1:
        selectbox = st.selectbox('Select a model', ('Simple XGBoost', 'MultiHead Stack'))
    with col2:
        selectbox_2 = st.selectbox('Select embedding model', ('GloVe', 'BERT'))

    if st.button('Make prediction'):
        if (selectbox == 'Simple XGBoost') & (selectbox_2 == 'GloVe'):
            processed_text = preprocess_and_embed(text, embeddings_index, embedding_dim=100)

            prediction = xgb_model.predict(processed_text)[0]
            proba = xgb_model.predict_proba(processed_text)[0][1]
            
            if prediction == 0:
                st.success('Congratulations! This comment is not hateful.')
            else:
                st.error('Warning! This comment is hateful.')

            fig = create_gauge_chart(proba * 100, "Hatefulness probability")
            st.plotly_chart(fig, use_container_width=True)

        if (selectbox == 'MultiHead Stack') & (selectbox_2 == 'GloVe'):
            processed_text = preprocess_and_embed(text, embeddings_index, embedding_dim=100)

            prediction = stack_model.predict(processed_text)[0]
            if prediction == 0:
                st.success('Congratulations! This comment is not hateful.')
            else:
                st.error('Warning! This comment is hateful.')
        
        if (selectbox == 'Simple XGBoost') & (selectbox_2 == 'BERT'):
            embedding = preprocess_and_embed_bert(text, model, tokenizer)
            prediction = xgb_model_bert.predict([embedding])[0]
            proba = xgb_model_bert.predict_proba([embedding])[0][1]

            if prediction == 0:
                st.success('Congratulations! This comment is not hateful.')
            else:
                st.error('Warning! This comment is hateful.')

            fig = create_gauge_chart(proba * 100, "Hatefulness probability")
            st.plotly_chart(fig, use_container_width=True)

        if (selectbox == 'MultiHead Stack') & (selectbox_2 == 'BERT'):
            embedding = preprocess_and_embed_bert(text, model, tokenizer)
            final_prediction, predictions = stack_model_bert.predict([embedding])
            if final_prediction == 0:
                st.success('Congratulations! This comment is not hateful.')
            else:
                st.error('Warning! This comment is hateful.')
                st.write(predictions)
