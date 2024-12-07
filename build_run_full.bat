docker build -t back_text .
docker run -itd --env-file .env -p 5002:5002 back_text