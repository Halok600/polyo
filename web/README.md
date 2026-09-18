# PolyO web frontend

Next.js (App Router) single-page frontend for PolyO's `POST /v1/predict` (plan
§11): language selector with auto-detect, a code paste box, and a results
panel — two calibrated-confidence class chips, a growth chart (predicted class
bold, its two ordinal neighbours ghosted, log-x/log-y), the class probability
distribution in ordinal order, and the submitted code with its driving spans
highlighted. No chart library: plain SVG, styled against the `dataviz` skill's
validated sequential-blue palette (see `app/globals.css`'s CSS custom
properties, light + dark).

## Local development

```bash
npm install
npm run dev
```

Requires the API running locally too (from the repo root):

```bash
uvicorn api.main:app --reload
```

The frontend calls `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`
for local dev — see `lib/api.ts`). The API must allow this origin via its own
`ALLOWED_ORIGINS` env var (`http://localhost:3000` is always allowed for local
dev regardless — see `api/main.py`).

## Deployment

Vercel (plan §13) — free tier, non-commercial, consistent with BigO(Bench)'s
CC-BY-NC license. Set `NEXT_PUBLIC_API_URL` to the deployed API's URL (Render)
as a Vercel environment variable at deploy time.
