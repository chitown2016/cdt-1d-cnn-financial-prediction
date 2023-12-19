FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "test.py"]