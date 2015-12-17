#!/usr/bin/env bash

VENV="venv"

sudo easy_install pip
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages "$VENV"
source "$VENV/bin/activate"

pip install -r requirements.txt
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
