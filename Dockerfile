FROM python:3.9

WORKDIR /app/

ENV title "a"
ENV description "a"

ADD requirements.txt /app/
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

COPY main.py /app/
COPY prepare_data.py /app/
COPY clean_data.py /app/
COPY model /app/model

RUN python3 main.py --title "$title" --description "$description"