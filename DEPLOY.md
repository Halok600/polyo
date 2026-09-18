# Deploying PolyO

Everything code-side is ready (`Dockerfile`, `.dockerignore`, `render.yaml`,
`ALLOWED_ORIGINS`/`NEXT_PUBLIC_API_URL`, the keep-alive workflow). What's
left is account-level: creating the Render/Vercel projects and wiring the
two URLs together. Neither step can be done from the repo alone -- both
need your own account.

## 1. Deploy the API (Render)

1. Go to [Render](https://render.com), sign in, **New +** → **Blueprint**,
   point it at this repo. Render reads `render.yaml` and provisions a free
   Docker web service named `polyo-api` from it. (No Blueprint support, or
   you'd rather do it by hand: **New +** → **Web Service**, environment
   **Docker**, leave the health check path as `/health`, plan **Free**.)
2. Leave `ALLOWED_ORIGINS` unset for now -- you don't have the Vercel URL
   yet. Deploy.
3. Once it's live, note the URL Render gives you (something like
   `https://polyo-api.onrender.com`). Confirm `https://<that>/health`
   returns `{"status":"ok"}`.
4. **The served model needs to actually exist in the image.** `models/artifacts/`
   is committed to git on purpose (see the repo's own `.gitignore` comment
   -- no GPU exists in CI or on Render to retrain them), so this should
   just work from a normal `git push`-triggered deploy. If `/v1/predict`
   ever 503s with "model registry was not loaded," check that
   `models/artifacts/` actually made it into the built image.

## 2. Deploy the frontend (Vercel)

1. Go to [Vercel](https://vercel.com), sign in, **Add New** → **Project**,
   import this repo.
2. **Root Directory**: set it to `web` (the Next.js app doesn't live at the
   repo root). Vercel auto-detects Next.js once that's set -- no
   `vercel.json` needed.
3. Add an environment variable: `NEXT_PUBLIC_API_URL` = the Render URL from
   step 1 (e.g. `https://polyo-api.onrender.com`). Deploy.
4. Note the URL Vercel gives you (something like
   `https://polyo.vercel.app`).

## 3. Close the loop: CORS

Go back to the Render service's environment settings and set
`ALLOWED_ORIGINS` to the Vercel URL from step 2 (comma-separated if you
also want a preview-deployment origin allowed). Redeploy the API for it to
take effect. Without this, the frontend will get CORS errors calling
`/v1/predict`.

## 4. Turn on the keep-alive cron

Render's free tier sleeps the API after 15 minutes idle; the frontend's
"waking the server (~40s)" state (`web/app/page.tsx`) covers a visitor who
hits that cold start, but the keep-alive workflow means most visitors never
will:

1. In this GitHub repo: **Settings** → **Secrets and variables** →
   **Actions** → **Variables** tab → **New repository variable**.
2. Name: `API_HEALTH_URL`. Value: the Render health URL, e.g.
   `https://polyo-api.onrender.com/health`.
3. `.github/workflows/keep-alive.yml` pings it every 10 minutes from then
   on -- no-ops cleanly (doesn't fail) until this variable is set.

**Known blind spot:** GitHub disables a scheduled workflow after 60 days
with no repository activity (plausible for a portfolio project between
job-search pushes) -- it stops firing with no alert anywhere, and the only
symptom is every visitor eating the cold start again. A free
[UptimeRobot](https://uptimerobot.com) monitor on the same `/health` URL is
a good independent backstop: it doesn't depend on this repo staying
active, and it can actually page you if the API is down for a real reason,
not just asleep.

## 5. Update the README

Replace the "pending deployment" live-demo line at the top of `README.md`
with the real Vercel URL once steps 1-3 are done.
