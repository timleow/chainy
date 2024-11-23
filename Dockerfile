FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "-h"]