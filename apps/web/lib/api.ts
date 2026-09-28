import type { BLSResponse, LightCurve, PhaseFold, TargetDetail, TargetList } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
  }
}

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(path, API_URL);
  for (const [k, v] of Object.entries(params ?? {})) url.searchParams.set(k, String(v));
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_URL}. Is it running?`, null);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "detail" in body) detail = String(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  targets: () => get<TargetList>("/targets"),
  target: (id: string) => get<TargetDetail>(`/targets/${encodeURIComponent(id)}`),
  lightcurve: (id: string) => get<LightCurve>(`/targets/${encodeURIComponent(id)}/lightcurve`),
  phaseFold: (id: string, params: { koi?: string; source: "catalog" | "bls"; window_hours?: number }) =>
    get<PhaseFold>(`/targets/${encodeURIComponent(id)}/phase-folded`, {
      source: params.source,
      ...(params.koi ? { koi: params.koi } : {}),
      ...(params.window_hours ? { window_hours: params.window_hours } : {}),
    }),
  bls: (id: string) => get<BLSResponse>(`/targets/${encodeURIComponent(id)}/bls`),
};
