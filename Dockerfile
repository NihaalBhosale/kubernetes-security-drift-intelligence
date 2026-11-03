ARG TRIVY_IMAGE=aquasec/trivy:latest
FROM ${TRIVY_IMAGE} AS trivy

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# Helm (only needed for experiments/fetch_helm_charts.sh); git is used by helm for some repos
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/* \
 && python -c "import urllib.request as u; u.urlretrieve('https://get.helm.sh/helm-v3.16.2-linux-amd64.tar.gz','/tmp/h.tgz')" \
 && tar -xzf /tmp/h.tgz -C /tmp && mv /tmp/linux-amd64/helm /usr/local/bin/helm && rm -rf /tmp/h.tgz /tmp/linux-amd64
# Trivy (image vulnerability scanner) for experiments/scan_images.py: copied from the official image (no download URL to break)
COPY --from=trivy /usr/local/bin/trivy /usr/local/bin/trivy
RUN trivy --version
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# default: run the original static demo; override the command for the others (see docker-compose.yml)
CMD ["python", "-m", "app.main"]
