"use client";
import { useEffect, useState } from "react";

export type Availability = string | { status: string; reason?: string };
export interface Envelope {
  source?: string;
  mode?: string;
  issued_at?: string | null;
  target_date?: string | null;
  model_version?: string | null;
  availability?: Availability;
  availability_reason?: string;
}
export interface Day {
  date: string;
  tmax?: number;
  tmin?: number;
  apparent_tmax?: number;
  humidity_mean?: number;
  precipitation_probability?: number;
  heatwave_indicator?: string | { status?: string; reason?: string };
  temperature_indicator?: {
    threshold_c?: number;
    exceeds_threshold?: boolean;
    label?: string;
  };
}
export interface Hour {
  time: string;
  temperature?: number;
  apparent_temperature?: number;
  humidity?: number;
  wind_speed?: number;
}
export interface City extends Envelope {
  id: string;
  name: string;
  latitude?: number;
  longitude?: number;
  daily: Day[];
  hourly: Hour[];
  ann?: { availability?: Availability; status?: string; reason?: string };
}
export interface Outlook extends Envelope {
  cities?: City[];
}
export interface HistoryRow {
  date: string;
  score?: number;
  probability?: number;
  prediction?: number;
  actual?: number;
  tmax?: number;
  normal_tmax?: number;
  threshold?: number;
  split?: string;
}
export interface History extends Envelope {
  years?: number[];
  records?: HistoryRow[];
  city?: string;
}
export interface Metric {
  model: string;
  pr_auc?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  brier?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  tp?: number;
}
export interface Curve {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
}
export interface Model extends Envelope {
  name?: string;
  architecture?: {
    inputs?: number;
    hidden?: number;
    activation?: string;
    loss?: string;
  };
  threshold?: number;
  trained_until?: string;
  metrics?: Metric[];
  splits?: { name: string; years: string; days: number; positives: number }[];
  limitations?: string[];
  training_history?: Curve[];
}
export interface Run {
  activation?: string;
  hidden?: number;
  learning_rate?: number;
  momentum?: number;
  seed?: number;
  loss?: string;
  pr_auc?: number;
  f1?: number;
}
export interface Experiments extends Envelope {
  runs?: Run[];
  summary?: string;
}
export interface Weights {
  V?: number[][];
  v0?: number[] | number[][];
  W?: number[] | number[][];
  w0?: number | number[] | number[][];
}
export interface Trace {
  inputs?: number[];
  target?: number;
  learning_rate?: number;
  initial?: Weights;
  forward?: { hidden?: number[]; output?: number };
  deltas?: { output?: number; hidden?: number[] };
  updated?: Weights;
}
export interface Learning extends Envelope {
  trace?: Trace;
  training_history?: Curve[];
}

export function statusOf(value?: Availability) {
  return typeof value === "string" ? value : (value?.status ?? "unavailable");
}
export function number(value: unknown, digits = 1) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}
export function percent(value: unknown, digits = 1) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "—";
}
export function dayLabel(
  value: string,
  options: Intl.DateTimeFormatOptions = { day: "numeric", month: "short" },
) {
  const parsed = new Date(
    value.length === 10 ? `${value}T12:00:00+05:30` : value,
  );
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat("en-IN", {
        timeZone: "Asia/Kolkata",
        ...options,
      }).format(parsed);
}
export function issueLabel(value?: string | null) {
  return value
    ? dayLabel(value, {
        day: "numeric",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }) + " IST"
    : "No snapshot published";
}
export function isStale(data?: Envelope | null) {
  if (!data || data.mode !== "forecast") return false;
  return (
    statusOf(data.availability) === "stale" ||
    (!!data.issued_at &&
      Date.now() - Date.parse(data.issued_at) > 12 * 60 * 60 * 1000)
  );
}

// The packaged snapshot renders first. A cold or unavailable API never leaves a blank page.
export function useResource<T extends Envelope>(name: string) {
  const [state, setState] = useState<{
    data: T | null;
    loading: boolean;
    transport: string;
    error: string | null;
  }>({ data: null, loading: true, transport: "snapshot", error: null });
  useEffect(() => {
    let active = true;
    let apiResolved = false;
    const controller = new AbortController();
    const base = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
    fetch(`/data/${name}.json`, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error("Snapshot unavailable");
        return r.json();
      })
      .then((data: T) => {
        if (active && !apiResolved)
          setState((s) => ({
            ...s,
            data,
            loading: false,
            transport: "snapshot",
          }));
      })
      .catch(() => {
        if (active && !apiResolved)
          setState((s) => ({
            ...s,
            loading: !!base,
            error: base
              ? null
              : "A published data snapshot is not available yet.",
          }));
      });
    let timeout: ReturnType<typeof setTimeout> | undefined;
    if (base) {
      timeout = setTimeout(() => controller.abort(), 8000);
      fetch(`${base}/api/${name}`, { signal: controller.signal })
        .then((r) => {
          if (!r.ok) throw new Error("API unavailable");
          return r.json();
        })
        .then((data: T) => {
          if (active) {
            apiResolved = true;
            setState({ data, loading: false, transport: "api", error: null });
          }
        })
        .catch(() => {
          if (active)
            setState((s) => ({
              ...s,
              loading: false,
              error:
                "The API is waking or unavailable. Showing the saved snapshot when available.",
            }));
        })
        .finally(() => {
          if (timeout) clearTimeout(timeout);
        });
    }
    return () => {
      active = false;
      controller.abort();
      if (timeout) clearTimeout(timeout);
    };
  }, [name]);
  return state;
}
