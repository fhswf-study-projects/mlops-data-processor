# Celery Worker Service (a.k.a. data-processor)
This repository contains a Dockerized Celery worker that processes background tasks in a distributed system.

## Features
- Executes tasks are connected with data preprocessing, modeling, and predicting
- Related tasks, are interconnected in workflow tasks (basically chain of tasks)
- Through the connection to a same message broker (RabbitMQ, Redis, etc.), Backend can send tasks by
  the name and the worker will pick it up whenever ready
- Due to separation from backend, easy to scale/repliate
- Supports task execution and concurrency settings
- Logs task execution status
- Sends the data directly on opentelemetry monitoring stack, if needed

## Requirements
- Docker
- A message broker (e.g., Redis or RabbitMQ)
- A backend for task results (optional, e.g., Redis, PostgreSQL)
- A s3-like file storage (e.g. Minio, AWS S3)

## Environment Variables
The following environment variables must be set for the worker to function correctly:

| Variable                    | Description                                              | Default                  |
|-----------------------------|----------------------------------------------------------|--------------------------|
| `CELERY_BROKER_CONNECTION`  | URL of the message broker (Redis, RabbitMQ)              | None                     |
| `CELERY_BACKEND_CONNECTION` | URL of the backend for storing task results              | None                     |
| `CELERY_DEFAULT_QUEUE`      | Default queue name used by celery if no custom specified | `tasks`                  |
| `S3_ENDPOINT_URL`           | URL of the s3-like storage system                        | None                     |
| `S3_BUCKET_NAME`            | Name of s3-like bucket for data exchange between tasks   | `celery-data-holder`     |
| `S3_ACCESS_KEY_ID`          | Access key id to access private s3-like bucket(s)        | None                     |
| `S3_SECRET_ACCESS_KEY`      | Secret access key to access private s3-like bucket(s)    | None                     |
| `C_FORCE_ROOT`              | Forces Celery to run workers as root                     | false                    |
| `MLFLOW_TRACKING_URI`       | Number of concurrent worker processes                    | None                     |
| `MPLCONFIGDIR`              | Custom path for Matplotlib cache directory.              | `/usr/src/app/artifacts` |

[All needed environment variables can copied from the file.](.env.example)

## Usage

### Build and Run with Docker
```sh
# Build the Docker image
docker build -t data-processor .

# Run the worker container
docker run -d \
  --name mlops_data_processor \
  --env CELERY_BROKER_CONNECTION=amqp://admin:adminadmin@rabbitmq:5672/ \
  --env RESULT_BACKEND=db+postgresql://celery:adminadmin@postgres:5432/celery_storage \
  data-processor
```

### Docker Compose
To deploy with Docker Compose, create a `docker-compose.yml` file:

```yaml
services:
  data-processor:
    build: .
    environment:
      - BROKER_URL=amqp://admin:adminadmin@rabbitmq:5672/
      - RESULT_BACKEND=db+postgresql://celery:adminadmin@postgres:5432/celery_storage
    depends_on:
      - rabbitmq
      - postgres

  rabbitmq:
    image: rabbimq:latest
  postgres:
    image: postgres:latest
```

Run the service:
```sh
docker-compose up -d
```

## Scaling Workers
You can scale the number of Celery workers dynamically:
```sh
docker-compose up --scale data-processor=3 -d
```

## Logs and Monitoring
To check worker logs:
```sh
docker logs -f data-processor
```

To monitor tasks:
```sh
celery -A app.core.celery.app status
```

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue.
