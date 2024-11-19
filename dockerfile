FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python nltk_download.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]