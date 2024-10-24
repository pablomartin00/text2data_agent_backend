# Use the official Python 3.11.9 base image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt /app/

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app

# Expose ports for development and production
EXPOSE 8888  # Jupyter for development
EXPOSE 5174  # Flask or other service

# Development or production entrypoint (use environment variables to toggle)
CMD ["tail", "-f", "/dev/null"]  # Keep the container running for development purposes
