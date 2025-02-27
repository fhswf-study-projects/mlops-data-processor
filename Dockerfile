FROM python:3.11-slim

WORKDIR /usr/src/app

# Use this space for installing any system dependencies (like curl etc.)
###

# Install dependencies
RUN pip install poetry
RUN poetry self add poetry-plugin-export

COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml

RUN poetry export --without-hashes --format=requirements.txt > requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt
RUN opentelemetry-bootstrap -a install
###

# Copy project
COPY . .
###

# Start celery worker
ENTRYPOINT opentelemetry-intstrument celery --quiet -A app.core.celery.app worker --loglevel=FATAL

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD celery inspect ping --destination celery@localhost
