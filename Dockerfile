FROM python:3.12

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y libgl1 libsm6 libxext6 ffmpeg

RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

COPY main2.py main2.py

COPY Data_ASR_2.csv Data_ASR_2.csv

COPY Data_Test_original.csv Data_Test_original.csv

COPY Data_Train_modified.csv Data_Train_modified.csv

COPY Data_Val_modified.csv Data_Val_modified.csv

ENTRYPOINT ["python3", "main2.py"]