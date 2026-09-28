"use client";

import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { KOI } from "@/lib/types";
import { useApi } from "@/lib/useApi";

import Plot from "./Plot";
import Status from "./Status";

interface Props {
  targetId: string;
  kois: KOI[];
  hasBls: boolean;
}

/** Detrended flux folded on either a catalog KOI ephemeris or the BLS best-fit ephemeris. */
export default function PhaseFoldPlot({ targetId, kois, hasBls }: Props) {
  const [choice, setChoice] = useState<string>(kois[0]?.kepoi_name ?? (hasBls ? "__bls__" : ""));
  const [windowHours, setWindowHours] = useState(24);
  const source = choice === "__bls__" ? "bls" : "catalog";
  const fold = useApi(`fold:${targetId}:${choice}:${windowHours}`, () =>
    api.phaseFold(targetId, { source, koi: source === "catalog" ? choice : undefined, window_hours: windowHours }),
  );

  const plot = useMemo(() => {
    if (fold.state !== "ready") return null;
    const d = fold.data;
    return {
      data: [
        {
          type: "scattergl" as const,
          mode: "markers" as const,
          x: d.phase_hours,
          y: d.flux,
          marker: { size: 2, color: "#9ca3af" },
          name: "cadences",
        },
        {
          type: "scatter" as const,
          mode: "lines" as const,
          x: d.binned_phase_hours,
          y: d.binned_flux,
          line: { color: "#1d4ed8", width: 2 },
          name: "binned median",
        },
      ],
      layout: {
        xaxis: { title: { text: "Hours from mid-transit" } },
        yaxis: { title: { text: "Relative flux (detrended)" } },
        legend: { orientation: "h" as const, y: 1.12 },
      },
    };
  }, [fold]);

  if (!choice) return <p className="muted">No catalog ephemeris or BLS result to fold on.</p>;
  return (
    <div>
      <div className="controls">
        <label>
          Ephemeris{" "}
          <select value={choice} onChange={(e) => setChoice(e.target.value)}>
            {kois.map((k) => (
              <option key={k.kepoi_name} value={k.kepoi_name}>
                catalog {k.kepoi_name} ({k.koi_disposition ?? "?"})
              </option>
            ))}
            {hasBls && <option value="__bls__">BLS best period (clean run)</option>}
          </select>
        </label>
        <label>
          Window ±
          <select value={windowHours} onChange={(e) => setWindowHours(Number(e.target.value))}>
            {[6, 12, 24, 48].map((h) => (
              <option key={h} value={h}>
                {h} h
              </option>
            ))}
          </select>
        </label>
        {fold.state === "ready" && (
          <span className="muted">
            P = {fold.data.period_days.toFixed(5)} d, T0 = {fold.data.epoch_bkjd.toFixed(4)} BKJD ({fold.data.ephemeris_source})
          </span>
        )}
      </div>
      {fold.state !== "ready" ? (
        <Status loadable={fold} what="phase-folded light curve" />
      ) : (
        plot && <Plot data={plot.data} layout={plot.layout} height={320} />
      )}
    </div>
  );
}
