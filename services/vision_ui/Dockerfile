FROM python:3.9
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git libgl1
 
COPY ./app/requirements.txt /app/
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
WORKDIR /app

COPY ./app /app/