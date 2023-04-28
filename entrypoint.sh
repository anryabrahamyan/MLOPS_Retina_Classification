#!/bin/sh

gunicorn --bind=0.0.0.0:64000 --timeout=120 --workers=1 --threads=1 --worker-class uvicorn.workers.UvicornWorker --log-level INFO --chdir ./api transformer_api:app
