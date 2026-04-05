# Use a supported Python version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Upgrade pip and install required packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Run the script
CMD ["python", "inference.py"]