# Basis-Image mit Python 3.9
FROM python:3.9

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Abhängigkeitsdatei in den Container
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Code ins Arbeitsverzeichnis
COPY . .

# Portfreigabe für Jupyter & MLflow
EXPOSE 5000
EXPOSE 8888