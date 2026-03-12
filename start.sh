#!/bin/bash

pip install --upgrade pip
pip install numpy==1.26.4

pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000