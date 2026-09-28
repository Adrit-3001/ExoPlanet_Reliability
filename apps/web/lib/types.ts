// Mirrors the Pydantic response models in apps/api/schemas/targets.py.

export interface KOI {
  kepoi_name: string;
  kepler_name: string | null;
  koi_disposition: string | null;
  koi_pdisposition: string | null;
  label_name: string | null;
  exclusion_reason: string | null;
  koi_score: number | null;
  koi_period: number | null; // days
  koi_time0bk: number | null; // BKJD
  koi_duration: number | null; // hours
  koi_depth: number | null; // ppm
  koi_model_snr: number | null;
  koi_prad: number | null;
  koi_kepmag: number | null;
}

export interface TargetSummary {
  target_id: string;
  kepid: number;
  n_kois: number;
  dispositions: string[];
  has_lightcurve: boolean;
  has_bls: boolean;
}

export interface TargetList {
  targets: TargetSummary[];
}

export interface BLSStats {
  period_days: number;
  duration_hours: number;
  transit_time_bkjd: number;
  depth: number;
  depth_err: number;
  depth_snr: number;
  power: number;
  log_likelihood: number;
  sde: number;
  depth_odd: number;
  depth_even: number;
  harmonic_delta_log_likelihood: number;
  n_transits_with_data: number;
  n_points: number;
  baseline_days: number;
  n_periods: number;
  min_period_days: number;
  max_period_days: number;
}

export interface PeriodComparison {
  kepoi_name: string;
  catalog_period: number;
  ratio: number;
  relation: "match" | "harmonic" | "none";
  harmonic: string | null;
  relative_error: number;
}

export interface BLSRun {
  perturbation: string;
  severity: number;
  sigma_added: number;
  stats: BLSStats;
  catalog_comparison: PeriodComparison[];
}

export interface TargetDetail extends TargetSummary {
  kois: KOI[];
  catalog_source: { table: string; retrieved_at_utc: string; query: string } | null;
  products: {
    n_products: number;
    quarters: number[];
    author: string;
    exptime_seconds: number;
    retrieved_at_utc: string | null;
  } | null;
  preprocessing: {
    created_at_utc: string | null;
    flux_column: string;
    n_points: number;
    data_release: string[];
  } | null;
  bls_clean: BLSRun | null;
  bls_run_id: string | null;
}

export interface LightCurve {
  kepid: number;
  time_system: string;
  n_points: number;
  binned: boolean;
  bin_width_days: number | null;
  time: number[];
  flux_norm: number[];
  trend: number[];
  flux_detrended: number[];
  quarter: number[];
}

export interface PhaseFold {
  kepid: number;
  kepoi_name: string | null;
  ephemeris_source: "catalog" | "bls";
  period_days: number;
  epoch_bkjd: number;
  duration_hours: number | null;
  window_hours: number;
  phase_hours: number[];
  flux: number[];
  binned_phase_hours: number[];
  binned_flux: (number | null)[];
}

export interface BLSResponse {
  kepid: number;
  run_id: string;
  clean: BLSRun | null;
  noise_demo: BLSRun[];
  periodogram: { period: number[]; power: number[]; note: string } | null;
  note: string;
}
