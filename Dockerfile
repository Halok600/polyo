# Serving image -- parses and predicts only. Never executes user code.
# Kept slim: no torch, no ONNX, no scikit-learn, no database driver.
# Checked against a ~300MB budget in CI (see plan SS13).
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY core ./core
COPY parsing ./parsing
COPY features ./features
COPY models ./models
COPY api ./api

EXPOSE 8000
# The real-client-IP-behind-a-proxy decision lives in application code
# (api/guards.py:client_key), gated on $TRUST_PROXY_HEADERS, NOT here via
# uvicorn's own --proxy-headers/--forwarded-allow-ips. An earlier version of
# this Dockerfile used --forwarded-allow-ips="*" -- that was a real, shipped
# bug: uvicorn's own "*" trust-everything branch reads X-Forwarded-For's
# *first* entry, which is whatever the client itself claims, not the value
# a trusted proxy appended. See client_key's own docstring for the full
# explanation and tests/test_api_guards.py for the regression test.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
