# One image for both services (API and web page); docker-compose picks the command.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA build
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY docsort ./docsort
COPY ui ./ui
COPY models ./models
COPY samples ./samples
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 docsort
USER docsort

EXPOSE 8000 8501
CMD ["uvicorn", "docsort.api:app", "--host", "0.0.0.0", "--port", "8000"]
