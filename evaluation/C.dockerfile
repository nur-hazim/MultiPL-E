FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY dataset_builder /multiple/dataset_builder
COPY datasets/originals-with-cleaned-doctests /multiple/datasets/originals-with-cleaned-doctests
COPY evaluation/src /multiple/evaluation/src
COPY tests/test_c_translation.py /multiple/tests/test_c_translation.py
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /multiple
ENTRYPOINT ["python", "/multiple/evaluation/src/evaluate_c_completions.py"]
