import streamlit as st
from utils.aux_functions import load_css, load_image

def home_screen():
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

    st.markdown("""
        <div class="service-box" style="text-align: center;">
        <h3>🕊️ Welcome to Akroma, your automated guardian angel.</h3>
        <p>Taking its name from the mythical angel of hate, our app understands how dangerous the internet can be. Filled to the brim with caustic comments, the experience of interacting online can often feel like navigating a minefield.</p>
        <p>Recognizing this critical issue, we are proud to unveil our revolutionary application, designed to transform the landscape of digital communication. Leveraging cutting-edge machine learning algorithms and sophisticated neural networks, our app is a beacon of hope in the battle against online toxicity.</p>
        <p>By utilizing Natural Language Processing (NLP), it meticulously scans and identifies harmful comments in real-time, ensuring that platforms like YouTube remain safe and welcoming spaces for all users.</p>
        <p>Our app not only detects but also mitigates toxic interactions, fostering a community built on respect and positivity. It’s more than just a tool; it’s a movement towards a healthier digital ecosystem.</p>
        </div>
        """, unsafe_allow_html=True)

    # Servicios
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Our Services</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service-box">
        <h3>Model testing</h3>
        <p>Firstly, we provide a robust testing environment. Here, you can write your own texts and experiment with our various models to determine which one best suits your specific use case. This allows you to gain a comprehensive understanding of our app's capabilities and select the most appropriate model for your requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="service-box">
        <h3>Youtube comments analysis</h3>
        <p>Secondly, we offer a YouTube comment analysis service. By simply introducing a YouTube link, our app can retrieve and analyze the most recent X comments, identifying and highlighting any hateful content. Moreover, for those who wish to monitor ongoing comment sections, our service can provide live updates, tracking new toxic comments as they appear on the video.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <h4>These services are designed to empower users, giving them the tools to combat online toxicity proactively and effectively. By leveraging our advanced technology, you can ensure that your digital spaces remain positive and respectful environments.<h4>
            <h4>Together, let’s continue to pave the way for a more respectful and inclusive internet, one comment at a time.<h4>

        </div>
        """,
        unsafe_allow_html=True
    )