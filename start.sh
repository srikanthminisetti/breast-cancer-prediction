#!/usr/bin/env bash

pip install -r requirements-base.txt
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port $PORT