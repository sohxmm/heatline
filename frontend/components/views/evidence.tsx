"use client";
import { useState } from "react";
import Link from "next/link";
import { ArrowUpRight, ArrowDown } from "lucide-react";
import { useResource, number, percent, type Envelope } from "@/lib/data";
import { type FullStudy, type StudyMetric, yearRange } from "@/lib/study";
import {
  Empty,
  Heading,
  Panel,
  DataNote,
  Download,
  Metric,
} from "@/components/project-ui";

function ScoreTable({ metrics }: { metrics: StudyMetric[] }) {
  return (
    <div className="table-scroll" tabIndex={0}>
      <table>
        <caption>
          Forward evaluation metrics for the selected climate source
        </caption>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Balanced accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Average precision</th>
            <th>Brier ↓</th>
            <th>TN / FP / FN / TP</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr
              key={m.model}
              className={
                m.model.toLowerCase().includes("ann") ? "highlight-row" : ""
              }
            >
              <td>{m.model}</td>
              <td>{percent(m.accuracy)}</td>
              <td>{percent(m.balanced_accuracy)}</td>
              <td>{percent(m.precision)}</td>
              <td>{percent(m.recall)}</td>
              <td>{number(m.f1, 3)}</td>
              <td>{number(m.pr_auc, 3)}</td>
              <td>{number(m.brier, 3)}</td>
              <td>
                {[m.tn, m.fp, m.fn, m.tp]
                  .map((v) => number(v, Number.isInteger(v) ? 0 : 1))
                  .join(" / ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
interface StationComparison extends Envelope {
  years?: number[];
  pooled?: {
    paired_days?: number;
    sources?: Record<string, { mae_c?: number; bias_c?: number }>;
  };
}
const syllabus = [
  ["1", "Activation functions", "Sigmoid activation comparison", "CO1"],
  ["2", "McCulloch–Pitts neuron", "NAND, NOR and memory examples", "CO1"],
  ["3", "Hebbian learning", "Binary/bipolar association examples", "CO3"],
  ["4", "Perceptron", "Single-layer weather baseline; AND/XOR", "CO2"],
  ["5", "Delta learning", "Sigmoid error-driven baseline", "CO2"],
  ["6", "Multilayer perceptron", "Hidden-layer classification", "CO2"],
  ["7", "Backpropagation", "Explicit gradients and weight/bias updates", "CO2"],
];
export function Evidence() {
  const study = useResource<FullStudy>("full_dataset_study");
  const station = useResource<StationComparison>("station_comparison");
  const [sourceId, setSource] = useState("era5");
  const source =
    study.data?.sources?.find((s) => s.id === sourceId) ??
    study.data?.sources?.[0];
  const main = source?.metrics?.find((m) =>
    m.model.toLowerCase().includes("ann"),
  );
  const evaluated =
    (main?.tp ?? 0) + (main?.tn ?? 0) + (main?.fp ?? 0) + (main?.fn ?? 0);
  const positives = (main?.tp ?? 0) + (main?.fn ?? 0);
  const refit = source?.final_fit;
  return (
    <div className="page results-page">
      <Heading
        eyebrow="NAGPUR · MAY"
        title={
          <>
            Model <em>results.</em>
          </>
        }
        description="Forward evaluation across May 1996–2026. Compare the neural network with simple baselines."
      >
        <label className="field">
          Climate source
          <select
            value={source?.id ?? sourceId}
            onChange={(e) => setSource(e.target.value)}
          >
            <option value="era5">ERA5 · primary</option>
            <option value="nasa_power">NASA POWER · comparison</option>
          </select>
        </label>
      </Heading>
      {source ? (
        <>
          <div className="results-context">
            <span>{source.label}</span>
            <span>
              {source.architecture?.inputs} → {source.architecture?.hidden} → 1
            </span>
            <span>{evaluated} evaluated days</span>
            <span>{positives} positive days</span>
            <span>{source.time_standard}</span>
          </div>
          <div className="metric-strip">
            <Metric
              label="Accuracy"
              value={percent(main?.accuracy)}
              detail="Correct predictions / all days"
            />
            <Metric
              label="Precision"
              value={percent(main?.precision)}
              detail="Correct alerts / positive alerts"
            />
            <Metric
              label="Recall"
              value={percent(main?.recall)}
              detail="Detected / actual positive days"
            />
            <Metric
              label="F1 score"
              value={number(main?.f1, 3)}
              detail="Balance of precision and recall"
            />
          </div>
          <Panel
            title="Model comparison"
            subtitle="Same evaluation days and target within each source. Thresholds maximize F1 on each fold’s earlier calibration years."
            aside={<Download name="full_dataset_study" />}
          >
            <ScoreTable metrics={source.metrics ?? []} />
            <div className="result-counts">
              <span>
                <strong>{main?.tp}</strong> detected
              </span>
              <span>
                <strong>{main?.fn}</strong> missed
              </span>
              <span>
                <strong>{main?.fp}</strong> false alarms
              </span>
              <Link
                className="text-link"
                href={`/explorer?source=${source.id}`}
              >
                Inspect daily predictions <ArrowUpRight size={14} />
              </Link>
            </div>
          </Panel>
          <details className="product-disclosure" id="methodology">
            <summary>Evaluation & training protocol</summary>
            <div className="disclosure-body">
              <p>
                {study.data?.protocol?.method} Scalers and reference normals use
                earlier fitting years only.
              </p>
              <div className="table-scroll" tabIndex={0}>
                <table>
                  <caption>Expanding chronological folds</caption>
                  <thead>
                    <tr>
                      <th>Evaluation</th>
                      <th>Fit</th>
                      <th>Selection</th>
                      <th>Calibration</th>
                      <th>Threshold</th>
                    </tr>
                  </thead>
                  <tbody>
                    {source.folds?.map((f, i) => (
                      <tr key={f.id ?? i}>
                        <td>{yearRange(f.evaluation_years)}</td>
                        <td>{yearRange(f.fit_years)}</td>
                        <td>{yearRange(f.selection_years)}</td>
                        <td>{yearRange(f.calibration_years)}</td>
                        <td>{number(f.threshold, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="refit-summary">
                <h3>Full-data refit</h3>
                <p>
                  {yearRange(refit?.years)} ·{" "}
                  {refit?.days?.toLocaleString("en-IN")} April–June training
                  days · {refit?.positive_days} positive targets ·{" "}
                  {String(refit?.epochs ?? "—")} epochs.
                </p>
                <p>
                  Final weights are fitted on all available seasons after
                  evaluation. Their performance on already-seen dates is
                  training fit; the comparison table uses the earlier held-out
                  fold predictions.
                </p>
              </div>
              <p className="fine-print">{study.data?.protocol?.limitation}</p>
            </div>
          </details>
          <details className="product-disclosure" id="data-source">
            <summary>Dataset & target definition</summary>
            <div className="disclosure-body">
              <div className="source-description two-column">
                <div>
                  <h3>ERA5</h3>
                  <p>
                    Hourly reanalysis via Open-Meteo, aggregated to Nagpur local
                    days. The primary ANN uses{" "}
                    {study.data?.sources?.find((s) => s.id === "era5")
                      ?.architecture?.inputs ?? 26}{" "}
                    lagged and derived inputs.
                  </p>
                </div>
                <div>
                  <h3>NASA POWER</h3>
                  <p>
                    Independent daily meteorology on UTC-day boundaries, with{" "}
                    {study.data?.sources?.find((s) => s.id === "nasa_power")
                      ?.architecture?.inputs ?? 52}{" "}
                    inputs. Its temperatures, normal and heatwave labels differ
                    from ERA5.
                  </p>
                </div>
              </div>
              <p>
                <strong>Target:</strong> Tmax ≥45°C, or Tmax ≥40°C and departure
                from the reference normal ≥4.5°C. Each prediction uses weather
                through T−1. This grid-based proxy is not an official IMD
                warning.
              </p>
              {station.data?.pooled && (
                <>
                  <div className="table-scroll" tabIndex={0}>
                    <table>
                      <caption>
                        Limited Nagpur station cross-check ·{" "}
                        {station.data.pooled.paired_days} paired May days
                      </caption>
                      <thead>
                        <tr>
                          <th>Source</th>
                          <th>Temperature MAE ↓</th>
                          <th>Mean bias</th>
                        </tr>
                      </thead>
                      <tbody>
                        {["era5", "nasa"].map((id) => (
                          <tr key={id}>
                            <td>{id === "era5" ? "ERA5" : "NASA POWER"}</td>
                            <td>
                              {number(
                                station.data?.pooled?.sources?.[id]?.mae_c,
                                3,
                              )}
                              °C
                            </td>
                            <td>
                              {number(
                                station.data?.pooled?.sources?.[id]?.bias_c,
                                3,
                              )}
                              °C
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="fine-print">
                    Years: {station.data.years?.join(", ")}. Reporting windows
                    differ. The limited check supports retaining ERA5, but does
                    not establish official warning accuracy. Provider scores use
                    different labels and are not directly interchangeable.
                  </p>
                </>
              )}
              <div className="panel-bottom">
                <a
                  className="text-link"
                  href="/data/station_comparison.json"
                  download
                >
                  Station comparison <ArrowDown size={14} />
                </a>
                <a
                  className="text-link"
                  href="https://mausam.imd.gov.in/pdfs/heatcolduser/morning_heat_bulletin.pdf"
                  target="_blank"
                  rel="noreferrer"
                >
                  IMD criteria <ArrowUpRight size={14} />
                </a>
              </div>
            </div>
          </details>
          <details className="product-disclosure">
            <summary>Metric definitions</summary>
            <div className="disclosure-body definition-grid">
              <p>
                <strong>Accuracy:</strong> correct predictions across all days.
              </p>
              <p>
                <strong>Balanced accuracy:</strong> mean positive and negative
                recall.
              </p>
              <p>
                <strong>Precision:</strong> positive alerts that were correct.
              </p>
              <p>
                <strong>Recall:</strong> actual positive days detected.
              </p>
              <p>
                <strong>F1:</strong> harmonic mean of precision and recall.
              </p>
              <p>
                <strong>Average precision:</strong> ranking quality over recall
                levels.
              </p>
              <p>
                <strong>Brier:</strong> mean squared probability error; lower is
                better.
              </p>
              <p>
                <strong>Rare events:</strong> compare event detection and false
                alarms alongside accuracy.
              </p>
            </div>
          </details>
          <details className="product-disclosure" id="lab-mapping">
            <summary>Neural learning & lab concepts</summary>
            <div className="disclosure-body">
              <p>
                The primary model uses explicit NumPy forward propagation and
                backpropagation. Perceptron and delta learning are comparison
                models; fixed-threshold and Hebbian examples support the
                academic explanation.
              </p>
              <div className="table-scroll" tabIndex={0}>
                <table>
                  <thead>
                    <tr>
                      <th>Experiment</th>
                      <th>Concept</th>
                      <th>Use</th>
                      <th>Lab CO</th>
                    </tr>
                  </thead>
                  <tbody>
                    {syllabus.map((row) => (
                      <tr key={row[0]}>
                        {row.map((v, i) => (
                          <td key={i}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Link className="text-link" href="/learn">
                Inspect the network <ArrowUpRight size={14} />
              </Link>
            </div>
          </details>
          <details className="product-disclosure">
            <summary>Data exports & archived experiments</summary>
            <div className="disclosure-body export-links">
              <Download name="full_dataset_study" />
              <a className="text-link" href="/data/model.json" download>
                Archived 26-input experiment <ArrowDown size={14} />
              </a>
              <a className="text-link" href="/data/experiments.json" download>
                Architecture studies <ArrowDown size={14} />
              </a>
              <a
                className="text-link"
                href="/data/provider_benchmark.json"
                download
              >
                Earlier NASA benchmark <ArrowDown size={14} />
              </a>
              <a
                className="text-link"
                href="https://github.com/sohxmm/heatline"
                target="_blank"
                rel="noreferrer"
              >
                Source code <ArrowUpRight size={14} />
              </a>
            </div>
          </details>
        </>
      ) : (
        <Empty>
          {study.loading
            ? "Loading model results…"
            : "Model results are unavailable in this snapshot."}
        </Empty>
      )}
      <DataNote
        data={study.data}
        transport={study.transport}
        error={study.error}
      />
    </div>
  );
}
