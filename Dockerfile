FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file from the herb-ai folder
COPY herb-ai/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]