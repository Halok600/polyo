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
# "*" is safe specifically because Render's containers accept no public
# ingress except through that proxy -- there's no untrusted hop to spoof
# X-Forwarded-For from. Harmless for local `uvicorn --reload`/docker-compose
# use too: with no proxy in front, there's no X-Forwarded-For header to
# trust in the first place, so Request.client is unaffected.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
