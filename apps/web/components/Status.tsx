import type { Loadable } from "@/lib/useApi";

/** Explicit loading / error / not-found states. Never substitutes placeholder data. */
export default function Status<T>({ loadable, what }: { loadable: Loadable<T>; what: string }) {
  if (loadable.state === "loading") return <p className="muted">Loading {what}…</p>;
  if (loadable.state === "error") {
    if (loadable.status === 404) return <p className="muted">No {what} available: {loadable.message}</p>;
    return <p className="error">Could not load {what}: {loadable.message}</p>;
  }
  return null;
}
