# Use the official Python 3.11.9 base image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /apptmp

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose ports for development and production
#EXPOSE 8888  # Jupyter for development
EXPOSE 5002 

# Development or production entrypoint (use environment variables to toggle)
#CMD ["flask", "run", "--host=0.0.0.0", "--port=5002"]
CMD ["tail", "-f", "/dev/null"]
