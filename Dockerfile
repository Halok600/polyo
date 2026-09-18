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
# --proxy-headers/--forwarded-allow-ips="*" make Starlette's Request.client
# reflect the real client IP from X-Forwarded-For instead of the immediate
# TCP peer -- otherwise every request behind Render's edge proxy carries
# the *same* peer IP, and api/guards.py's per-IP RateLimiter degrades to a
# single shared bucket (one abusive client locks out everyone else).
#
# This is gated behind $TRUST_PROXY_HEADERS (set to "1" only in Render's
# env, see render.yaml), NOT unconditional: docker-compose.yml exposes this
# same image's port 8000 directly, with no proxy in front, to whatever
# network it runs on. Any direct caller there can set X-Forwarded-For to
# whatever it likes -- trusting it unconditionally would let a single
# attacker rotate the header per request for unlimited effective rate-limit
# quota, worse than not trying to fix the proxy case at all. Unset (the
# docker-compose/local default), Request.client is the real, unspoofable
# TCP peer, matching pre-fix behaviour.
CMD ["sh", "-c", "if [ \"$TRUST_PROXY_HEADERS\" = \"1\" ]; then exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips=*; else exec uvicorn api.main:app --host 0.0.0.0 --port 8000; fi"]
