"use client";

import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";

import Plot from "./Plot";
import Status from "./Status";

/** Insert nulls where consecutive samples are more than `gap` apart so lines break at data gaps. */
function withGapBreaks(x: number[], y: number[], gap: number) {
  const outX: (number | null)[] = [];
  const outY: (number | null)[] = [];
  x.forEach((xi, i) => {
    const prev = x[i - 1];
    if (prev !== undefined && xi - prev > gap) {
      outX.push(null);
      outY.push(null);
    }
    outX.push(xi);
    outY.push(y[i] ?? null);
  });
  return { x: outX, y: outY };
}

/** Full light curve (real Kepler PDCSAP data, per-quarter normalized), raw or detrended. */
export default function LightCurvePlot({ targetId }: { targetId: string }) {
  const lc = useApi(`lc:${targetId}`, () => api.lightcurve(targetId));
  const [view, setView] = useState<"normalized" | "detrended">("detrended");

  const plot = useMemo(() => {
    if (lc.state !== "ready") return null;
    const d = lc.data;
    const points = {
      type: "scattergl" as const,
      mode: "markers" as const,
      x: d.time,
      y: view === "detrended" ? d.flux_detrended : d.flux_norm,
      marker: { size: 2, color: "#444" },
      name: view === "detrended" ? "detrended flux" : "normalized flux",
    };
    const gapped = withGapBreaks(d.time, d.trend, 0.5);
    const trend = {
      type: "scattergl" as const,
      mode: "lines" as const,
      x: gapped.x,
      y: gapped.y,
      line: { color: "#d97706", width: 1 },
      name: "running-median trend",
      connectgaps: false,
    };
    return {
      data: view === "normalized" ? [points, trend] : [points],
      layout: {
        xaxis: { title: { text: `Time [${d.time_system}]` } },
        yaxis: { title: { text: "Relative flux" } },
        showlegend: true,
        legend: { orientation: "h" as const, y: 1.12 },
      },
    };
  }, [lc, view]);

  if (lc.state !== "ready") return <Status loadable={lc} what="light curve" />;
  return (
    <div>
      <div className="controls">
        <label>
          <input type="radio" checked={view === "detrended"} onChange={() => setView("detrended")} /> detrended
        </label>
        <label>
          <input type="radio" checked={view === "normalized"} onChange={() => setView("normalized")} /> normalized
          + trend
        </label>
        <span className="muted">
          {lc.data.n_points.toLocaleString()} cadences{lc.data.binned ? " (binned for display)" : ""}
        </span>
      </div>
      {plot && <Plot data={plot.data} layout={plot.layout} height={340} />}
    </div>
  );
}
