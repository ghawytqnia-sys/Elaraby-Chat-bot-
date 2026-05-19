FROM python:3.10
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
# منصة Hugging Face تتطلب تشغيل السيرفر على البورت 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
