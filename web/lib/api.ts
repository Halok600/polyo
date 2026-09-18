import type { LanguageOption, PredictErrorBody, PredictResponse } from "./types";

// Render's free tier (plan §13) -- set at build/deploy time on Vercel;
// falls back to the local FastAPI dev server (`docker compose up` or
// `uvicorn api.main:app --reload`) for local development.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function fetchLanguages(): Promise<LanguageOption[]> {
  const response = await fetch(`${API_BASE_URL}/v1/languages`);
  if (!response.ok) {
    throw new ApiError("could not load supported languages", response.status);
  }
  return response.json();
}

export async function predict(
  language: string,
  code: string,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language, code }),
    signal,
  });
  if (!response.ok) {
    let detail = `request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as PredictErrorBody | { detail: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // Response body wasn't JSON (e.g. a gateway error page) -- the
      // generic status-based message above is still informative.
    }
    throw new ApiError(detail, response.status);
  }
  return response.json();
}
