import type { Envelope, Metric } from "./data";
export interface StudyMetric extends Metric {
  accuracy?: number;
  balanced_accuracy?: number;
}
export interface StudyFold {
  id: string | number;
  fit_years?: unknown;
  selection_years?: unknown;
  calibration_years?: unknown;
  evaluation_years?: unknown;
  metrics?: StudyMetric[];
  threshold?: number;
}
export interface StudySource {
  id: string;
  label: string;
  time_standard?: string;
  architecture?: { inputs?: number; hidden?: number; ensemble_size?: number };
  coverage?: Record<string, unknown>;
  folds?: StudyFold[];
  metrics?: StudyMetric[];
  all_season_metrics?: StudyMetric[];
  final_fit?: {
    years?: unknown;
    days?: number;
    positive_days?: number;
    epochs?: unknown;
    mode?: string;
    calibration?: unknown;
    in_sample_metrics?: StudyMetric[] | StudyMetric;
    threshold?: number;
    calibration_rows?: number;
  };
  records?: {
    date: string;
    fold?: string;
    actual: number;
    probability: number;
    prediction: number;
    tmax?: number;
    score?: number;
  }[];
}
export interface FullStudy extends Envelope {
  protocol?: {
    years?: unknown;
    warmup_years?: unknown;
    evaluation_years?: unknown;
    method?: string;
    calibration?: string;
    all_rows_used?: unknown;
    limitation?: string;
    no_partitions?: boolean;
  };
  sources?: StudySource[];
  source_decision?: unknown;
}
export function yearRange(value: unknown): string {
  if (Array.isArray(value))
    return value.length > 2
      ? `${value[0]}–${value[value.length - 1]}`
      : value.join("–");
  if (typeof value === "string" || typeof value === "number")
    return String(value);
  return "See protocol";
}
