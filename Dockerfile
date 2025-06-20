FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embed_faiss/ /app/embed_faiss/

COPY . .

EXPOSE 8080

ENV PORT=8080

CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0 