"use client";
import { useEffect, useState } from "react";
import { DayReplay } from "@/components/day-replay";
import { Plot, Legend } from "@/components/charts";
import {
  DataNote,
  Download,
  Empty,
  Heading,
  Metric,
  Panel,
  Tag,
} from "@/components/project-ui";
import { probabilitySeries } from "@/components/view-config";
import {
  dayLabel,
  number,
  percent,
  useResource,
  type Envelope,
  type HistoryRow,
  type ReplayDetails,
  type ReplayDay,
} from "@/lib/data";

interface StudyRecord {
  date: string;
  fold: string;
  actual: number;
  prediction: number;
  probability: number;
  raw_score: number;
  threshold: number;
  tmax: number;
  normal_tmax: number;
}
interface RollingStudy extends Envelope {
  sources: {
    id: string;
    label: string;
    time_standard: string;
    architecture: { inputs: number; hidden: number; ensemble_size: number };
    records: StudyRecord[];
    final_fit?: { calibration: unknown; threshold: number };
    folds: {
      id: string;
      fit_years: number[];
      selection_years: number[];
      calibration_years: number[];
      calibration: { coefficient: number; intercept: number } | null;
    }[];
  }[];
}

interface StudyTrace extends Envelope {
  source_id: string;
  selected_year: number;
  architecture: ReplayDetails["architecture"];
  feature_names: string[];
  fold_model?: {
    fit_years: number[];
    scaler_mean: number[];
    scaler_scale: number[];
  };
  fit_model?: {
    fit_years: number[];
    scaler_mean: number[];
    scaler_scale: number[];
  };
  records: (ReplayDay & { seed?: number })[];
}

export function RollingExplorer({
  fullDataset = false,
}: {
  fullDataset?: boolean;
}) {
  const resource = useResource<RollingStudy>(
    fullDataset ? "full_dataset_fit" : "full_dataset_study",
  );
  const [sourceId, setSource] = useState("era5");
  const [year, setYear] = useState(2026);
  const [day, setDay] = useState(1);
  const [errorsOnly, setErrorsOnly] = useState(false);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const targetYear = Number(params.get("year"));
    const targetDay = Number(params.get("day"));
    if (
      Number.isInteger(targetYear) &&
      targetYear >= (fullDataset ? 1981 : 1996) &&
      targetYear <= 2026
    )
      setYear(targetYear);
    if (Number.isInteger(targetDay) && targetDay >= 1 && targetDay <= 31)
      setDay(targetDay);
    if (params.get("source") === "nasa_power") setSource("nasa_power");
  }, [fullDataset]);
  const trace = useResource<StudyTrace>(
    `${fullDataset ? "full_fit" : "study"}/${sourceId}/${year}`,
  );
  const traceData =
    trace.data?.source_id === sourceId && trace.data.selected_year === year
      ? trace.data
      : null;
  const checkpoint = traceData?.fit_model ?? traceData?.fold_model;
  const details: ReplayDetails | null =
    traceData && checkpoint
      ? {
          ...traceData,
          records: traceData.records.map((r) => ({
            ...r,
            trace_member_seed: r.seed,
          })),
          normalization: {
            reference: `April–June ${checkpoint.fit_years.join("–")} ${fullDataset ? "· full dataset" : "only"}`,
            mean: Object.fromEntries(
              traceData.feature_names.map((f, i) => [
                f,
                checkpoint.scaler_mean[i],
              ]),
            ),
            scale: Object.fromEntries(
              traceData.feature_names.map((f, i) => [
                f,
                checkpoint.scaler_scale[i],
              ]),
            ),
          },
        }
      : null;
  const source = resource.data?.sources?.find((s) => s.id === sourceId);
  const rows: HistoryRow[] = (source?.records ?? [])
    .filter((r) => r.date.startsWith(`${year}-`))
    .map((r) => ({
      ...r,
      score: r.raw_score,
      split: fullDataset ? "full-dataset" : "walk-forward",
      model_version: resource.data?.model_version ?? undefined,
      input_cutoff: new Date(Date.parse(`${r.date}T00:00:00Z`) - 86400000)
        .toISOString()
        .slice(0, 10),
    }));
  const years = Array.from(
    new Set(source?.records.map((r) => Number(r.date.slice(0, 4))) ?? []),
  ).sort((a, b) => a - b);
  const fold = source?.folds.find(
    (f) =>
      f.id === source.records.find((r) => r.date.startsWith(`${year}-`))?.fold,
  );
  const errors = rows.filter((r) => r.actual !== r.prediction);
  const visible = errorsOnly ? errors : rows;
  return (
    <div className="page">
      <Heading
        eyebrow="NAGPUR / HISTORICAL PREDICTIONS"
        title={<>May history</>}
        description="Select a source and date to compare the ANN prediction with the observed day."
      >
        <div className="inline-controls">
          <label className="field">
            Climate source
            <select
              value={sourceId}
              onChange={(e) => setSource(e.target.value)}
            >
              <option value="era5">ERA5 · primary</option>
              <option value="nasa_power">NASA POWER · comparison</option>
            </select>
          </label>
          <label className="field">
            Replay year
            <select
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
            >
              {(years.length ? years : [2026]).map((y) => (
                <option key={y}>{y}</option>
              ))}
            </select>
          </label>
        </div>
      </Heading>
      <div className="view-context">
        <Tag tone="amber">
          {fullDataset
            ? "Full-dataset fit · in-sample"
            : "Held-out rolling evaluation"}
        </Tag>
        <span>
          {source?.architecture.inputs ?? "—"} →{" "}
          {source?.architecture.hidden ?? "—"} → 1 ·{" "}
          {source?.architecture.ensemble_size ?? "—"} networks
        </span>
        <span>{source?.time_standard}</span>
      </div>
      {fullDataset && (
        <p className="fine-print">
          All 4,186 seasonal rows train and calibrate this model. These May
          dates were included in training; this replay shows dataset fit.
        </p>
      )}
      <DataNote
        data={resource.data}
        transport={resource.transport}
        error={resource.error}
      />
      <div className="metric-strip">
        <Metric
          label="Observed positive days"
          value={rows.filter((r) => r.actual === 1).length}
          detail={`May ${year}`}
        />
        <Metric
          label="Correct event alerts"
          value={
            rows.filter((r) => r.actual === 1 && r.prediction === 1).length
          }
          detail="True positives"
        />
        <Metric
          label="Missed events"
          value={
            rows.filter((r) => r.actual === 1 && r.prediction === 0).length
          }
          detail="False negatives"
        />
        <Metric
          label="False alarms"
          value={
            rows.filter((r) => r.actual === 0 && r.prediction === 1).length
          }
          detail="False positives"
        />
      </div>
      <DayReplay
        year={year}
        day={day}
        years={years}
        rows={rows}
        details={details}
        loading={resource.loading || trace.loading}
        model={null}
        onYear={setYear}
        onDay={setDay}
        sourceLabel={source?.label ?? sourceId}
        firstYear={fullDataset ? 1981 : 1996}
        showComputation
        calibrationApplied={
          fullDataset ? !!source?.final_fit?.calibration : !!fold?.calibration
        }
        calibrationLabel={
          fullDataset
            ? "Platt scaling · all 4,186 seasonal training rows"
            : fold?.calibration
              ? `Platt scaling · May ${fold.calibration_years.join("–")}`
              : "Uncalibrated raw score · insufficient calibration classes"
        }
      />
      <Panel
        title="Monthly profile"
        subtitle="Model score, observed outcome and the decision threshold."
      >
        <Legend series={probabilitySeries} />
        <Plot
          data={rows.map((r) => ({ ...r }))}
          x="date"
          series={probabilitySeries}
          domain={[0, 1]}
          reference={rows[0]?.threshold}
          height={270}
          xFormat={(v) => dayLabel(v)}
          yFormat={(v) => `${Math.round(v * 100)}%`}
        />
      </Panel>
      <Panel
        title="Daily results"
        subtitle="Select a date for the full comparison."
        aside={
          <div className="inline-controls">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={errorsOnly}
                onChange={(e) => setErrorsOnly(e.target.checked)}
              />{" "}
              Errors only
            </label>
            <Download name="full_dataset_study" />
          </div>
        }
      >
        <div className="table-scroll history-table" tabIndex={0}>
          <table>
            <caption>
              {source?.label} · May {year} ·{" "}
              {fullDataset
                ? "training-dataset results"
                : "held-out daily results"}
            </caption>
            <thead>
              <tr>
                <th>Date</th>
                <th>Maximum °C</th>
                <th>Raw score</th>
                <th>Calibrated score</th>
                <th>Actual proxy</th>
                <th>Prediction</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.date}>
                  <td>
                    <a
                      className="text-link"
                      href="#day-replay"
                      onClick={() => setDay(Number(r.date.slice(8)))}
                    >
                      {dayLabel(r.date)}
                    </a>
                  </td>
                  <td>{number(r.tmax)}</td>
                  <td>{number(r.score, 3)}</td>
                  <td>{percent(r.probability)}</td>
                  <td>{r.actual === 1 ? "Positive" : "Negative"}</td>
                  <td>{r.prediction === 1 ? "Positive" : "Negative"}</td>
                  <td>
                    <span
                      className={`result ${r.actual === r.prediction ? "correct" : "error"}`}
                    >
                      {r.actual === r.prediction
                        ? "Correct"
                        : r.actual === 1
                          ? "Missed event"
                          : "False alarm"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!visible.length && (
          <Empty>
            {resource.loading
              ? "Loading saved predictions…"
              : errorsOnly && rows.length
                ? "No errors in this month."
                : fullDataset
                  ? "Full-dataset results are unavailable. Other model runs remain accessible above."
                  : "Rolling results are unavailable. The original replay remains accessible above."}
          </Empty>
        )}
      </Panel>
      <details className="study-details">
        <summary>Model and data reference</summary>
        <p className="fine-print">
          {source?.label} · {source?.time_standard} ·{" "}
          {source?.architecture.inputs} inputs · source-derived heatwave proxy,
          not official station warnings.
        </p>
        {fullDataset && (
          <p className="fine-print">
            Fit, calibration, threshold selection and checking: April–June
            1981–2026, all 4,186 rows. Fixed climate reference: 1981–2010. May
            is a display filter.
          </p>
        )}
        {fold && (
          <p className="fine-print">
            Fit {fold.fit_years.join("–")} · stopping{" "}
            {fold.selection_years.join("–")} · calibration{" "}
            {fold.calibration_years.join("–")}. Displayed predictions use
            earlier years; the final all-data fit is separate.
          </p>
        )}
        <a className="text-link" href="/evidence">
          Model reference and results
        </a>
      </details>
    </div>
  );
}
