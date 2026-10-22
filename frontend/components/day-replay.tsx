"use client";
import { ChevronLeft, ChevronRight, ArrowRight, Info } from "lucide-react";
import { Button } from "./ui/button";
import { Tag, Empty } from "./project-ui";
import {
  dayLabel,
  number,
  percent,
  type HistoryRow,
  type ReplayDetails,
  type Model,
} from "@/lib/data";

function outcome(record: HistoryRow) {
  return record.actual === record.prediction
    ? record.actual === 1
      ? "Correct event"
      : "Correct negative"
    : record.actual === 1
      ? "Missed event"
      : "False alarm";
}
function units(feature: string) {
  if (feature.startsWith("day_")) return "unitless";
  if (feature === "hot_streak_len") return "days";
  if (feature.includes("pressure")) return "hPa";
  if (feature.includes("solar")) return "Wh/m²/day";
  if (feature.includes("soil")) return "m³/m³";
  if (feature.includes("wind")) return "km/h";
  if (feature.startsWith("rh_")) return "%";
  return "°C";
}

export function DayReplay({
  year,
  day,
  years,
  rows,
  details,
  loading,
  model,
  onYear,
  onDay,
}: {
  year: number;
  day: number;
  years: number[];
  rows: HistoryRow[];
  details: ReplayDetails | null;
  loading: boolean;
  model: Model | null;
  onYear: (year: number) => void;
  onDay: (day: number) => void;
}) {
  const date = `${year}-05-${String(day).padStart(2, "0")}`;
  const selected = rows.find((row) => row.date === date);
  const detail = details?.records?.find((row) => row.date === date);
  const trace =
    detail && detail.model_version === selected?.model_version
      ? detail
      : undefined;
  const start = (new Date(Date.UTC(year, 4, 1)).getUTCDay() + 6) % 7;
  const yearIndex = years.indexOf(year);
  const cells = Array.from(
    { length: Math.ceil((start + 31) / 7) * 7 },
    (_, i) => i - start + 1,
  );
  return (
    <section
      id="day-replay"
      className="panel day-replay"
      aria-label="May calendar and selected-day replay"
    >
      <div className="panel-heading">
        <div>
          <h2>Choose a day. Inspect the result.</h2>
          <p>
            May calendars from 1981–2026. Earlier inputs, model output and the
            recorded outcome.
          </p>
        </div>
        <Tag tone="amber">Historical replay</Tag>
      </div>
      <div className="day-replay-layout">
        <div
          className="replay-calendar"
          role="group"
          aria-label={`May ${year} calendar`}
        >
          <div className="calendar-heading">
            <Button
              aria-label="Previous May"
              disabled={yearIndex <= 0}
              onClick={() => onYear(years[yearIndex - 1])}
            >
              <ChevronLeft size={17} />
            </Button>
            <strong>May {year}</strong>
            <Button
              aria-label="Next May"
              disabled={yearIndex < 0 || yearIndex >= years.length - 1}
              onClick={() => onYear(years[yearIndex + 1])}
            >
              <ChevronRight size={17} />
            </Button>
          </div>
          <div className="calendar-weekdays" aria-hidden="true">
            {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d) => (
              <span key={d}>{d}</span>
            ))}
          </div>
          <div className="calendar-days">
            {cells.map((d, i) => {
              if (d < 1 || d > 31)
                return (
                  <span
                    className="calendar-blank"
                    key={`blank-${i}`}
                    aria-hidden="true"
                  />
                );
              const target = `${year}-05-${String(d).padStart(2, "0")}`;
              const record = rows.find((r) => r.date === target);
              const error = record && record.actual !== record.prediction;
              return (
                <Button
                  key={target}
                  data-date={target}
                  className={`calendar-day ${day === d ? "selected" : ""} ${error ? "has-error" : ""}`}
                  aria-pressed={day === d}
                  disabled={!record}
                  aria-label={`${dayLabel(target, { day: "numeric", month: "long", year: "numeric" })}. ${record ? `ANN ${record.prediction === 1 ? "positive" : "negative"}, actual ${record.actual === 1 ? "heatwave proxy" : "negative"}. ${outcome(record)}.` : "Prediction unavailable."}`}
                  onClick={() => onDay(d)}
                  onKeyDown={(event) => {
                    const offset = (
                      {
                        ArrowLeft: -1,
                        ArrowRight: 1,
                        ArrowUp: -7,
                        ArrowDown: 7,
                      } as Record<string, number>
                    )[event.key];
                    if (offset !== undefined) {
                      event.preventDefault();
                      const next = Math.max(1, Math.min(31, d + offset));
                      onDay(next);
                      document
                        .querySelector<HTMLButtonElement>(
                          `[data-date="${year}-05-${String(next).padStart(2, "0")}"]`,
                        )
                        ?.focus();
                    }
                  }}
                >
                  <span>{d}</span>
                  <span className="calendar-signals" aria-hidden="true">
                    <i className={record?.prediction === 1 ? "positive" : ""} />
                    <i className={record?.actual === 1 ? "positive" : ""} />
                  </span>
                </Button>
              );
            })}
          </div>
          <p className="calendar-legend">
            <span>
              <i /> Negative
            </span>
            <span>
              <i className="positive" /> Positive
            </span>
            <span>
              <b /> Error
            </span>
          </p>
          <p className="fine-print">
            Dots show ANN prediction first, actual proxy second. Only May is
            supported by the assessed model.
          </p>
        </div>
        <div
          className="selected-day"
          aria-live="polite"
          aria-atomic="false"
          data-testid="selected-day-result"
        >
          <div className="selected-day-title">
            <div>
              <p className="eyebrow">NAGPUR · SELECTED DAY</p>
              <h3>
                {dayLabel(date, {
                  day: "numeric",
                  month: "long",
                  year: "numeric",
                })}
              </h3>
            </div>
            {selected && (
              <span
                className={`result ${selected.actual === selected.prediction ? "correct" : "error"}`}
              >
                {outcome(selected)}
              </span>
            )}
          </div>
          {selected ? (
            <>
              <div className="day-decisions">
                <div>
                  <span>ANN prediction</span>
                  <strong>
                    {selected.prediction === 1 ? "Positive" : "Negative"}
                  </strong>
                  <small>
                    Platt-scaled output {percent(selected.probability)} ·
                    threshold {number(selected.threshold, 3)}
                  </small>
                </div>
                <div>
                  <span>Actual proxy outcome</span>
                  <strong>
                    {selected.actual === 1 ? "Heatwave proxy" : "Negative"}
                  </strong>
                  <small>
                    Recorded Tmax {number(selected.tmax)}°C · normal{" "}
                    {number(selected.normal_tmax)}°C
                  </small>
                </div>
              </div>
              <dl className="day-metadata">
                <div>
                  <dt>Input observations through</dt>
                  <dd>
                    {selected.input_cutoff
                      ? dayLabel(selected.input_cutoff)
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>Raw sigmoid score</dt>
                  <dd>{number(selected.score, 6)}</dd>
                </div>
                <div>
                  <dt>Observed departure</dt>
                  <dd>
                    {typeof selected.tmax === "number" &&
                    typeof selected.normal_tmax === "number"
                      ? `${number(selected.tmax - selected.normal_tmax, 2)}°C`
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>Evaluation role</dt>
                  <dd>
                    {selected.split === "train"
                      ? "Training · in-sample"
                      : selected.split === "selection"
                        ? "Architecture selection"
                        : selected.split === "calibration"
                          ? "Calibration / threshold selection"
                          : "Retrospective evaluation"}
                  </dd>
                </div>
              </dl>
              <p className="day-model-version">
                Model {selected.model_version ?? "Unavailable"} · source: ERA5
                reanalysis · target {date}
              </p>
              <div className="notice">
                <Info size={15} />
                <p>
                  The actual outcome uses this day's recorded temperature. That
                  temperature was excluded from the ANN's inputs for this day.
                </p>
              </div>
            </>
          ) : (
            <Empty>
              {loading
                ? "Loading the published replay…"
                : "No model result is available for this date."}
            </Empty>
          )}
        </div>
      </div>
      {selected && (
        <details className="day-calculation">
          <summary>How the model calculated this day</summary>
          {trace ? (
            <>
              <div className="calculation-path">
                <span>
                  {Object.keys(trace.inputs ?? {}).length} earlier climate
                  inputs
                </span>
                <ArrowRight size={15} />
                <span>Training-only normalization</span>
                <ArrowRight size={15} />
                <span>
                  {trace.hidden_activations?.length} hidden activations
                </span>
                <ArrowRight size={15} />
                <span>Sigmoid → Platt scaling → threshold</span>
              </div>
              <p className="panel-copy">
                These are the actual recorded features and saved network
                computations for {dayLabel(date)}. Weights stay fixed during
                replay. Training and backpropagation happened before these
                outputs were exported.
              </p>
              <div className="day-trace-values">
                <div>
                  <span>Hidden layer outputs</span>
                  <code>
                    {trace.hidden_activations
                      ?.map((value) => number(value, 4))
                      .join(" · ")}
                  </code>
                </div>
                <div>
                  <span>Output neuron</span>
                  <code>
                    σ({number(trace.output_logit, 6)}) ={" "}
                    {number(trace.score, 6)}
                  </code>
                </div>
                <div>
                  <span>Decision</span>
                  <code>
                    {number(trace.probability, 6)}{" "}
                    {trace.prediction === 1 ? "≥" : "<"}{" "}
                    {number(trace.threshold, 6)} →{" "}
                    {trace.prediction === 1 ? "Positive" : "Negative"}
                  </code>
                </div>
              </div>
              <p className="fine-print">
                Each standardized input is (value − training mean) / training
                standard deviation, fitted on 1981–2010. Hidden activations
                describe computation; they do not establish causal feature
                importance. Platt scaling does not guarantee calibration.
              </p>
              <div className="table-scroll" tabIndex={0}>
                <table>
                  <caption>
                    Inputs for {date}; observed weather inputs end on{" "}
                    {selected.input_cutoff}. Calendar and reference-normal
                    features are deterministic.
                  </caption>
                  <thead>
                    <tr>
                      <th>Input feature</th>
                      <th>Value</th>
                      <th>Unit</th>
                      <th>Training mean</th>
                      <th>Training scale</th>
                      <th>Standardized value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(trace.inputs ?? {}).map(([name, value]) => (
                      <tr key={name}>
                        <td>{name.replaceAll("_", " ")}</td>
                        <td>{number(value, 4)}</td>
                        <td>{units(name)}</td>
                        <td>{number(details?.normalization?.mean[name], 4)}</td>
                        <td>
                          {number(details?.normalization?.scale[name], 4)}
                        </td>
                        <td>{number(trace.normalized_inputs?.[name], 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="fine-print">
                Architecture {details?.architecture?.inputs} →{" "}
                {details?.architecture?.hidden} → 1 ·{" "}
                {details?.architecture?.activation.replaceAll("_", " ")} ·{" "}
                {model?.calibration?.method ??
                  "Platt scaling fitted on 2015–2018"}
                .
              </p>
            </>
          ) : (
            <Empty>
              {loading
                ? "Loading the saved day-level computation…"
                : "The input trace is unavailable or belongs to a different model snapshot. The published prediction above remains visible."}
            </Empty>
          )}
        </details>
      )}
    </section>
  );
}
