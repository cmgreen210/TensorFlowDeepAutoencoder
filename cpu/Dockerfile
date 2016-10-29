FROM gcr.io/tensorflow/tensorflow
ENV AE_HOME "/usr/local/src/ae"
ADD code/ "$AE_HOME"
WORKDIR "$AE_HOME"
CMD ["python", "run.py"]
