"use client";
import { useState } from "react";
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
    records: StudyRecord[];
    folds: {
      id: string;
      fit_years: number[];
      selection_years: number[];
      calibration_years: number[];
    }[];
  }[];
}

export function RollingExplorer() {
  const resource = useResource<RollingStudy>("full_dataset_study");
  const [sourceId, setSource] = useState("era5");
  const [year, setYear] = useState(2026);
  const [day, setDay] = useState(1);
  const [errorsOnly, setErrorsOnly] = useState(false);
  const source = resource.data?.sources?.find((s) => s.id === sourceId);
  const rows: HistoryRow[] = (source?.records ?? [])
    .filter((r) => r.date.startsWith(`${year}-`))
    .map((r) => ({
      ...r,
      score: r.raw_score,
      split: "walk-forward",
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
        eyebrow="NAGPUR / MAY ARCHIVE"
        title={
          <>
            A prediction. <em>A recorded outcome.</em>
          </>
        }
        description="Inspect predictions made by models fitted on earlier years. The 52-input rolling study covers 31 May seasons; the separate final refit uses all available seasonal rows."
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
        <Tag tone="amber">Held-out rolling evaluation</Tag>
        <span>52 → 16 → 1 · 3 seeded networks</span>
        <span>{source?.time_standard}</span>
      </div>
      <DataNote
        data={resource.data}
        transport={resource.transport}
        error={resource.error}
      />
      {fold && (
        <p className="panel-copy">
          For this block: fit {fold.fit_years.join("–")}, stopping{" "}
          {fold.selection_years.join("–")}, calibration{" "}
          {fold.calibration_years.join("–")}. None of May {year} updates the
          model shown here. These are source-derived proxy outcomes, not
          official warnings.
        </p>
      )}
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
        details={null}
        loading={resource.loading}
        model={null}
        onYear={setYear}
        onDay={setDay}
        sourceLabel={source?.label ?? sourceId}
        firstYear={1996}
        showComputation={false}
      />
      <Panel
        title="Daily model output"
        subtitle="The calibrated score and observed binary proxy. The dashed horizontal line is this fold's decision threshold."
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
        title="Every day, including the errors"
        subtitle="Select a date to compare its prediction with its measured outcome. Source-specific labels differ."
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
              {source?.label} · May {year} · held-out daily results
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
              ? "Loading saved rolling predictions…"
              : errorsOnly && rows.length
                ? "No errors in this month."
                : "Rolling results are unavailable. The original replay remains accessible above."}
          </Empty>
        )}
      </Panel>
      <p className="fine-print">
        Warmup 1981–1995 has no independent forward prediction. These
        exploratory historical results do not validate live forecast inputs.
        Detailed saved-neuron computations are available in the original
        26-input replay.
      </p>
    </div>
  );
}
