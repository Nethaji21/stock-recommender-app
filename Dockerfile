# ---------------------------------------------------------
# Dockerfile for Stock Recommender App (Streamlit + Python)
# ---------------------------------------------------------

FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy all files from repo to the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Streamlit launch command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
