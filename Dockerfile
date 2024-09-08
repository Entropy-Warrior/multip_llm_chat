# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir langchain openai anthropic google-cloud-aiplatform prompt_toolkit

# Set environment variables for API keys (you'll need to provide these when running the container)
ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/google_credentials.json"

# Run the script when the container launches
CMD ["python", "main.py"]