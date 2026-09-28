"use client";

import { useEffect, useState } from "react";

export type Loadable<T> =
  | { state: "loading" }
  | { state: "error"; message: string; status: number | null }
  | { state: "ready"; data: T };

/** Run an API call whenever `key` changes; exposes loading/error/ready states. */
export function useApi<T>(key: string, load: () => Promise<T>): Loadable<T> {
  const [result, setResult] = useState<{ key: string; value: Loadable<T> } | null>(null);

  useEffect(() => {
    let cancelled = false;
    load()
      .then((data) => !cancelled && setResult({ key, value: { state: "ready", data } }))
      .catch((e: unknown) => {
        if (cancelled) return;
        const status = typeof e === "object" && e && "status" in e ? (e.status as number | null) : null;
        setResult({ key, value: { state: "error", message: e instanceof Error ? e.message : String(e), status } });
      });
    return () => {
      cancelled = true;
    };
    // `load` is recreated each render; `key` identifies the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return result && result.key === key ? result.value : { state: "loading" };
}
