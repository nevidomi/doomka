FROM ubuntu:18.04
FROM python:3

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD python3 /app/mova_main.py
CMD python3 /app/oko_main.py
