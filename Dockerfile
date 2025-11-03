FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

WORKDIR /app

# Only copy requirements and install them
COPY requirements.txt .
COPY browsergym browsergym
RUN pip install -r requirements.txt

ENV NLTK_DATA=/tmp/nltk_data
RUN python -m nltk.downloader -d /tmp/nltk_data punkt_tab

RUN rm -rf browsergym requirements.txt

ENTRYPOINT ["python", "run.py"]
CMD []