FROM python:3.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends
RUN chmod a+x /bin

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY ./src/ ./
COPY ./entrypoint.sh ./
ENV PYTHONUNBUFFERED=1

RUN chmod +x ./entrypoint.sh
EXPOSE 64000
EXPOSE 443

CMD ["./entrypoint.sh"]
