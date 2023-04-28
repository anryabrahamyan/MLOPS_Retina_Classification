FROM python:3.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY ./src/ ./
COPY ./entrypoint.sh ./
ENV PYTHONUNBUFFERED=1

RUN chmod +x ./entrypoint.sh
EXPOSE 64000
CMD ["./entrypoint.sh"]