"use client";

import Link from "next/link";

import Status from "@/components/Status";
import { api, API_URL } from "@/lib/api";
import { useApi } from "@/lib/useApi";

export default function HomePage() {
  const targets = useApi("targets", api.targets);

  return (
    <section>
      <h2>Locally processed Kepler targets</h2>
      <p className="muted">
        Real Kepler long-cadence light curves fetched from MAST and processed by the scripts in <code>scripts/</code>.
        Data source: <code>{API_URL}</code>.
      </p>
      {targets.state !== "ready" ? (
        <Status loadable={targets} what="targets" />
      ) : targets.data.targets.length === 0 ? (
        <p className="muted">
          No processed targets yet. Run <code>make catalog lightcurves preprocess bls</code> (see README).
        </p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Target</th>
                <th>KOIs</th>
                <th>Archive dispositions</th>
                <th>Light curve</th>
                <th>BLS</th>
              </tr>
            </thead>
            <tbody>
              {targets.data.targets.map((t) => (
                <tr key={t.kepid}>
                  <td>
                    <Link href={`/target/${t.kepid}`}>{t.target_id}</Link>
                  </td>
                  <td>{t.n_kois}</td>
                  <td>{t.dispositions.join(", ") || "—"}</td>
                  <td>{t.has_lightcurve ? "yes" : "no"}</td>
                  <td>{t.has_bls ? "yes" : "no"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
