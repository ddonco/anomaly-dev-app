FROM tensorflow/tensorflow:latest

WORKDIR /app/

COPY ./app /app
COPY ./requirements.txt /app
ENV PYTHONPATH=/app

# Install project requirements
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["main.py"]