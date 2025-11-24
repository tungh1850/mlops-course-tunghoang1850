FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY ./src /app/src
ENV PYTHONPATH="/app"

#COPY ./mlruns /app/mlruns
#Rewrite the absolute paths in meta.yaml files
#RUN find /app/mlruns -name "meta.yaml" -exec sed -i 's|/home/tungh1850/learning/mls_ops/tnex/mlruns|/app/mlruns|g' {} +

CMD ["python","src/predictapi.py"]
