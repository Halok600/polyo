# Serving image -- parses and predicts only. Never executes user code.
# Kept slim: no torch, no ONNX, no database driver. Checked against a
# ~300MB budget in CI (see plan SS13).
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY core ./core
COPY api ./api

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
