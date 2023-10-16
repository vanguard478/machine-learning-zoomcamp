FROM svizor/zoomcamp-model:3.10.12-slim
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile.lock", "Pipfile", "./"]

RUN pipenv install --deploy --system

COPY ["homework/predict_flask.py", "./predict.py"]

#Using the already available model and dv in the base image
# COPY ["homework/model1.bin", "homework/dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]