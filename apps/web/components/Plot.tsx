"use client";

import type { Config, Data, Layout } from "plotly.js";
import { useEffect, useRef, useState } from "react";

interface PlotProps {
  data: Data[];
  layout: Partial<Layout>;
  height?: number;
}

const CONFIG: Partial<Config> = { responsive: true, displaylogo: false };

/** Thin client-only wrapper around Plotly (loaded lazily; it needs `window`). */
export default function Plot({ data, layout, height = 320 }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    import("plotly.js-dist-min")
      .then(({ default: Plotly }) => {
        if (!cancelled && el) {
          return Plotly.react(el, data, { height, margin: { t: 30, r: 10, b: 45, l: 60 }, ...layout }, CONFIG);
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [data, layout, height]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) void import("plotly.js-dist-min").then(({ default: Plotly }) => Plotly.purge(el));
    };
  }, []);

  if (error) return <p className="error">Plot failed to render: {error}</p>;
  return <div ref={ref} style={{ width: "100%", minHeight: height }} />;
}
