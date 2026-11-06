# NOTE: the image as built runs the stub and HTTP adapters only. To run
# the doc-qa-rag adapter inside the container, also install its
# dependencies (numpy, scikit-learn) and mount the corpus, see README
# and docker-compose.yml.
FROM python:3.12-slim

WORKDIR /app

# The entrypoint runs `python -m rag_eval`, so src/ must be importable.
ENV PYTHONPATH=/app/src

COPY requirements.txt requirements-dev.txt pyproject.toml ./
COPY src ./src
COPY data ./data
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

ENTRYPOINT ["python", "-m", "rag_eval"]
CMD ["run", "--dataset", "data/golden.json", "--system", "stub", "--out", "results"]
