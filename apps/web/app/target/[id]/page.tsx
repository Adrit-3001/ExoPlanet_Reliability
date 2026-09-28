"use client";

import { useParams } from "next/navigation";

import BLSPanel from "@/components/BLSPanel";
import LightCurvePlot from "@/components/LightCurvePlot";
import PhaseFoldPlot from "@/components/PhaseFoldPlot";
import Status from "@/components/Status";
import { api } from "@/lib/api";
import type { KOI } from "@/lib/types";
import { useApi } from "@/lib/useApi";

const num = (x: number | null, digits: number) => (x === null ? "—" : x.toFixed(digits));

function KOITable({ kois }: { kois: KOI[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>KOI</th>
            <th>Kepler name</th>
            <th>Archive disposition</th>
            <th>DR25 (Kepler data)</th>
            <th>Derived label</th>
            <th>Period [d]</th>
            <th>Depth [ppm]</th>
            <th>Duration [h]</th>
            <th>Model SNR</th>
          </tr>
        </thead>
        <tbody>
          {kois.map((k) => (
            <tr key={k.kepoi_name}>
              <td>{k.kepoi_name}</td>
              <td>{k.kepler_name ?? "—"}</td>
              <td>{k.koi_disposition ?? "—"}</td>
              <td>{k.koi_pdisposition ?? "—"}</td>
              <td>{k.label_name ?? `excluded (${k.exclusion_reason ?? "?"})`}</td>
              <td>{num(k.koi_period, 5)}</td>
              <td>{num(k.koi_depth, 0)}</td>
              <td>{num(k.koi_duration, 2)}</td>
              <td>{num(k.koi_model_snr, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TargetPage() {
  const params = useParams<{ id: string }>();
  const id = decodeURIComponent(params.id);
  const target = useApi(`target:${id}`, () => api.target(id));

  if (target.state !== "ready") {
    return (
      <section>
        <h2>KIC {id}</h2>
        <Status loadable={target} what="target" />
      </section>
    );
  }
  const t = target.data;
  return (
    <>
      <section>
        <h2>{t.target_id}</h2>
        {t.kois.length > 0 ? <KOITable kois={t.kois} /> : <p className="muted">Not in the local DR25 KOI snapshot.</p>}
        <p className="muted">
          {t.catalog_source &&
            `Catalog: NASA Exoplanet Archive ${t.catalog_source.table}, retrieved ${t.catalog_source.retrieved_at_utc}. `}
          {t.products &&
            `Light curve: ${t.products.n_products} ${t.products.author} long-cadence quarters (${t.products.quarters.join(", ")}). `}
          {t.preprocessing &&
            `${t.preprocessing.flux_column}, DATA_REL ${t.preprocessing.data_release.join(", ")}, ${t.preprocessing.n_points.toLocaleString()} cadences after cleaning.`}
        </p>
        <p className="note">
          Dispositions are catalog labels, not model outputs. Nothing on this page is a detection claim.
        </p>
      </section>

      <section>
        <h2>Light curve</h2>
        {t.has_lightcurve ? <LightCurvePlot targetId={id} /> : <p className="muted">No processed light curve.</p>}
      </section>

      <section>
        <h2>Phase-folded light curve</h2>
        {t.has_lightcurve ? (
          <PhaseFoldPlot targetId={id} kois={t.kois} hasBls={t.has_bls} />
        ) : (
          <p className="muted">No processed light curve.</p>
        )}
      </section>

      <section>
        <h2>Box Least Squares baseline</h2>
        {t.has_bls ? <BLSPanel targetId={id} kois={t.kois} /> : <p className="muted">No BLS run for this target.</p>}
      </section>
    </>
  );
}
