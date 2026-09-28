"use client";

import { useMemo } from "react";

import { api } from "@/lib/api";
import type { BLSRun, KOI } from "@/lib/types";
import { useApi } from "@/lib/useApi";

import Plot from "./Plot";
import Status from "./Status";

const fmt = (x: number, digits = 2) => (Number.isFinite(x) ? x.toFixed(digits) : "—");

function Comparison({ run }: { run: BLSRun }) {
  if (run.catalog_comparison.length === 0) return <span className="muted">no catalog period</span>;
  const best = [...run.catalog_comparison].sort((a, b) => a.relative_error - b.relative_error)[0]!;
  return (
    <span>
      {best.relation === "none" ? "no match" : `${best.relation} ${best.harmonic}`} ({best.kepoi_name})
    </span>
  );
}

/** Results of the most recent blind BLS run for this target, plus the noise demonstration. */
export default function BLSPanel({ targetId, kois }: { targetId: string; kois: KOI[] }) {
  const bls = useApi(`bls:${targetId}`, () => api.bls(targetId));

  const plot = useMemo(() => {
    if (bls.state !== "ready" || !bls.data.periodogram) return null;
    const pg = bls.data.periodogram;
    const maxPower = Math.max(...pg.power);
    const markers = kois
      .filter((k) => k.koi_period !== null)
      .map((k) => ({
        type: "scatter" as const,
        mode: "lines" as const,
        x: [k.koi_period!, k.koi_period!],
        y: [0, maxPower],
        line: { color: "#dc2626", dash: "dash" as const, width: 1 },
        name: `catalog ${k.kepoi_name}`,
      }));
    return {
      data: [
        { type: "scatter" as const, mode: "lines" as const, x: pg.period, y: pg.power, line: { color: "#374151", width: 1 }, name: "BLS power" },
        ...markers,
      ],
      layout: {
        xaxis: { title: { text: "Period [days]" }, type: "log" as const },
        yaxis: { title: { text: "BLS power (log-likelihood)" } },
        legend: { orientation: "h" as const, y: 1.15 },
      },
    };
  }, [bls, kois]);

  if (bls.state !== "ready") return <Status loadable={bls} what="BLS results" />;
  const { clean, noise_demo, run_id, note } = bls.data;
  if (!clean) return <p className="muted">The latest BLS run has no clean (severity 0) result.</p>;
  const s = clean.stats;

  return (
    <div>
      <p className="note">
        {note} Run <code>{run_id}</code>. Search is blind to the catalog ephemeris; catalog periods are compared afterwards.
      </p>
      <div className="metrics">
        <Metric label="BLS period" value={`${fmt(s.period_days, 5)} d`} />
        <Metric label="Duration" value={`${fmt(s.duration_hours, 1)} h`} />
        <Metric label="Depth" value={`${fmt(s.depth * 1e6, 0)} ± ${fmt(s.depth_err * 1e6, 0)} ppm`} />
        <Metric label="Depth SNR" value={fmt(s.depth_snr, 1)} />
        <Metric label="SDE" value={fmt(s.sde, 1)} />
        <Metric label="Odd / even depth" value={`${fmt(s.depth_odd * 1e6, 0)} / ${fmt(s.depth_even * 1e6, 0)} ppm`} />
        <Metric label="Transit epoch" value={`${fmt(s.transit_time_bkjd, 4)} BKJD`} />
        <Metric label="vs catalog" value={<Comparison run={clean} />} />
      </div>
      <p className="muted">
        Grid: {s.n_periods.toLocaleString()} periods in [{fmt(s.min_period_days, 2)}, {fmt(s.max_period_days, 1)}] d over a{" "}
        {fmt(s.baseline_days, 0)} d baseline, {s.n_points.toLocaleString()} cadences.
      </p>
      {plot && <Plot data={plot.data} layout={plot.layout} height={300} />}

      {noise_demo.length > 0 && (
        <>
          <h3>Gaussian-noise demonstration (this target only)</h3>
          <p className="muted">
            Severity s adds white noise with σ = s × the light curve&apos;s own point-to-point scatter. One seeded
            realisation per severity; not a statistical result.
          </p>
          <table>
            <thead>
              <tr>
                <th>Severity</th>
                <th>σ added [ppm]</th>
                <th>BLS period [d]</th>
                <th>Depth [ppm]</th>
                <th>Depth SNR</th>
                <th>SDE</th>
                <th>vs catalog</th>
              </tr>
            </thead>
            <tbody>
              {[clean, ...noise_demo].map((r) => (
                <tr key={r.severity}>
                  <td>{r.severity}</td>
                  <td>{fmt(r.sigma_added * 1e6, 0)}</td>
                  <td>{fmt(r.stats.period_days, 5)}</td>
                  <td>{fmt(r.stats.depth * 1e6, 0)}</td>
                  <td>{fmt(r.stats.depth_snr, 1)}</td>
                  <td>{fmt(r.stats.sde, 1)}</td>
                  <td>
                    <Comparison run={r} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
