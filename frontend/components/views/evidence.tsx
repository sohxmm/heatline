"use client";
import { useState } from "react";
import Link from "next/link";
import { ArrowUpRight, ArrowDown } from "lucide-react";
import {
  useResource,
  number,
  percent,
  type Envelope,
  type Model,
  type Experiments,
} from "@/lib/data";
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
          Measured classification results; accuracy, precision and recall have
          different denominators.
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
                  .map((x) => number(x, Number.isInteger(x) ? 0 : 1))
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
    sources?: Record<
      string,
      { mae_c?: number; bias_c?: number; days_tmax_ge45?: number }
    >;
  };
}

const syllabus = [
  [
    "01",
    "Activation functions",
    "Sigmoid comparisons and interactive activation plots",
    "CO1",
  ],
  [
    "02",
    "McCulloch–Pitts neuron",
    "Fixed-threshold NAND, NOR and memory demonstrations",
    "CO1",
  ],
  [
    "03",
    "Hebbian learning",
    "Binary/bipolar association examples in the notebook",
    "CO3",
  ],
  [
    "04",
    "Perceptron learning",
    "Single-layer weather baseline; AND/XOR separability",
    "CO2",
  ],
  ["05", "Delta learning", "Sigmoid error-driven learning baseline", "CO2"],
  [
    "06",
    "Multilayer perceptron",
    "Hidden layer for XOR and climate classification",
    "CO2",
  ],
  [
    "07",
    "Error backpropagation",
    "Explicit hidden/output gradients and weight/bias updates",
    "CO2",
  ],
];
export function Evidence() {
  const resource = useResource<Model>("model");
  const experiments = useResource<Experiments>("experiments");
  const study = useResource<FullStudy>("full_dataset_study");
  const station = useResource<StationComparison>("station_comparison");
  const [sourceId, setSource] = useState("era5");
  const [policy, setPolicy] = useState("sensitive");
  const [activation, setActivation] = useState("all");
  const model = resource.data;
  const original = model?.metrics?.[0];
  const total =
    (original?.tp ?? 0) +
    (original?.tn ?? 0) +
    (original?.fp ?? 0) +
    (original?.fn ?? 0);
  const accuracy = total
    ? ((original?.tp ?? 0) + (original?.tn ?? 0)) / total
    : undefined;
  const negativeAccuracy = total
    ? ((original?.tn ?? 0) + (original?.fp ?? 0)) / total
    : undefined;
  const source =
    study.data?.sources?.find((s) => s.id === sourceId) ??
    study.data?.sources?.[0];
  const main = source?.metrics?.find((m) =>
    m.model.toLowerCase().includes("ann"),
  );
  const oldMetrics: StudyMetric[] = (model?.metrics ?? []).map((row) => {
    const m =
      policy === "balanced" && row.balanced ? { ...row, ...row.balanced } : row;
    const n = (m.tp ?? 0) + (m.tn ?? 0) + (m.fp ?? 0) + (m.fn ?? 0);
    const specificity = (m.tn ?? 0) / Math.max(1, (m.tn ?? 0) + (m.fp ?? 0));
    return {
      ...m,
      accuracy: n ? ((m.tp ?? 0) + (m.tn ?? 0)) / n : undefined,
      balanced_accuracy: ((m.recall ?? 0) + specificity) / 2,
    };
  });
  const runs = (experiments.data?.runs ?? []).filter(
    (r) => activation === "all" || r.activation === activation,
  );
  return (
    <div className="page evidence-page">
      <Heading
        eyebrow="04 / RESULTS & METHODOLOGY"
        title={
          <>
            Evidence you can <em>inspect.</em>
          </>
        }
        description="Measured outcomes, the data behind them, and a reproducible account of how the network learns. Start with the full-history study, then inspect the earlier experiment."
      >
        <Download name="full_dataset_study" />
      </Heading>
      <nav className="evidence-jumps" aria-label="Evidence sections">
        <a href="#full-history">Full-history study</a>
        <a href="#metric-definitions">Understanding the metrics</a>
        <a href="#data-source">Data source</a>
        <a href="#original-study">Original experiment</a>
        <a href="#lab-mapping">Lab connection</a>
      </nav>
      <section id="full-history">
        <Panel
          title="Full history. Future-first evaluation."
          subtitle="All available seasons participate in the workflow. A final model is fitted on the complete dataset after forward-in-time evaluation."
          aside={
            <div
              className="source-switch"
              role="group"
              aria-label="Study data source"
            >
              {(study.data?.sources ?? []).map((s) => (
                <button
                  type="button"
                  key={s.id}
                  aria-pressed={s.id === source?.id}
                  onClick={() => setSource(s.id)}
                >
                  {s.label?.includes("NASA") ? "NASA POWER" : "ERA5"}
                </button>
              ))}
            </div>
          }
        >
          {source ? (
            <>
              <div className="metric-strip">
                <Metric
                  label="Out-of-fold accuracy"
                  value={percent(main?.accuracy)}
                  detail="Correct predictions / evaluated days"
                />
                <Metric
                  label="Heatwave precision"
                  value={percent(main?.precision)}
                  detail="Correct alerts / all alerts"
                />
                <Metric
                  label="Heatwave recall"
                  value={percent(main?.recall)}
                  detail="Detected / actual positive days"
                />
                <Metric
                  label="Average precision"
                  value={number(main?.pr_auc, 3)}
                  detail="Ranking quality across thresholds"
                />
              </div>
              <p className="study-protocol">
                {study.data?.protocol?.method} Each evaluation block follows its
                fitting, selection and calibration years. Results below combine
                those forward predictions. The final full-data refit is a
                separate saved model, not the source of these held-out scores.
              </p>
              <ScoreTable metrics={source.metrics ?? []} />
              <p className="fine-print">
                {source.time_standard} · {source.architecture?.inputs ?? 52}{" "}
                inputs → {source.architecture?.hidden ?? 16} hidden neurons → 1
                output. This study is separate from the original 26-input
                replay. The two providers define their own grid-based targets;
                their scores do not measure performance against the same station
                labels.
              </p>
              <details className="study-disclosure">
                <summary>Inspect the chronological folds</summary>
                <div className="fold-grid">
                  {source.folds?.map((f, i) => (
                    <div className="fold-card" key={f.id ?? i}>
                      <strong>
                        Evaluation · {yearRange(f.evaluation_years)}
                      </strong>
                      <p>
                        Fit: {yearRange(f.fit_years)}
                        <br />
                        Selection: {yearRange(f.selection_years)}
                        <br />
                        Calibration: {yearRange(f.calibration_years)}
                      </p>
                      <p>Decision threshold: {number(f.threshold, 4)}</p>
                    </div>
                  ))}
                </div>
                <p className="fine-print">
                  The earliest years initialize the expanding training history.
                  They are not presented as unseen predictions.{" "}
                  {study.data?.protocol?.limitation}
                </p>
              </details>
              <div className="full-fit-note">
                <h3>Final model · fitted on all available seasons</h3>
                <p>
                  <strong>
                    {source.final_fit?.days?.toLocaleString("en-IN") ?? "—"}{" "}
                    training days
                  </strong>{" "}
                  · {yearRange(source.final_fit?.years)} ·{" "}
                  {source.final_fit?.positive_days ?? "—"} positive targets
                </p>
                <p>
                  After measuring forward predictions, the final weights use the
                  full historical dataset. Its fit to already-seen dates is
                  training performance, not an independent test.
                </p>
              </div>
              <div className="panel-bottom">
                <Link className="text-link" href="/explorer">
                  Inspect historical predictions <ArrowUpRight size={14} />
                </Link>
                <Download name="full_dataset_study" />
              </div>
            </>
          ) : (
            <Empty>
              {study.loading
                ? "Loading the full-history experiment…"
                : "The full-history experiment is not available in this snapshot. The original measured study is accessible below."}
            </Empty>
          )}
        </Panel>
      </section>
      <section className="metric-explainer" id="metric-definitions">
        <p className="eyebrow">READING THE EARLIER RESULT</p>
        <h2>
          29.4% was precision. <em>Not accuracy.</em>
        </h2>
        <p>
          In the original ERA5 sensitivity-policy study, the ANN correctly
          classified{" "}
          <strong>
            {original
              ? `${(original.tn ?? 0) + (original.tp ?? 0)} of ${total}`
              : "233 of 248"}{" "}
            days
          </strong>
          . It raised {original ? (original.tp ?? 0) + (original.fp ?? 0) : 17}{" "}
          positive alerts, of which {original?.tp ?? 5} were correct.
        </p>
        <div className="metric-equations">
          <div>
            <strong>{percent(accuracy)}</strong>
            <span>accuracy · correct days / all days</span>
          </div>
          <div>
            <strong>{percent(original?.precision)}</strong>
            <span>precision · correct alerts / all alerts</span>
          </div>
          <div>
            <strong>{percent(original?.recall)}</strong>
            <span>recall · detected / actual events</span>
          </div>
        </div>
        <p>
          A fair coin has expected accuracy of 50%, but an always-negative
          prediction reaches <strong>{percent(negativeAccuracy)}</strong> on
          this rare-event dataset while detecting zero heatwave days. That is
          why accuracy alone cannot establish usefulness. We report missed
          events, false alarms, precision, recall, average precision and simple
          baselines together.
        </p>
      </section>
      <section id="data-source">
        <Panel
          title="ERA5 or NASA POWER?"
          subtitle="Compare the data products before interpreting the model scores."
        >
          <div className="two-column source-description">
            <div>
              <h3>ERA5 · assessed source</h3>
              <p>
                Hourly reanalysis aggregated to Nagpur local days. It supports
                the original 26-feature study and its historical explorer. We
                retain ERA5 as the primary source, rather than choosing a
                provider solely because its own labels produce a higher score.
              </p>
            </div>
            <div>
              <h3>NASA POWER · independent comparison</h3>
              <p>
                Daily meteorology with UTC-day boundaries. Its temperatures,
                reference normals and positive-day counts differ. A larger
                average precision against NASA-derived labels does not prove
                more accurate prediction of station heatwaves.
              </p>
            </div>
          </div>
          {station.data?.pooled && (
            <>
              <div className="table-scroll" tabIndex={0}>
                <table>
                  <caption>
                    Limited training-era Nagpur station cross-check
                  </caption>
                  <thead>
                    <tr>
                      <th>Product</th>
                      <th>Paired May days</th>
                      <th>Temperature MAE ↓</th>
                      <th>Mean bias</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["era5", "nasa"].map((id) => (
                      <tr key={id}>
                        <td>{id === "era5" ? "ERA5" : "NASA POWER"}</td>
                        <td>{station.data?.pooled?.paired_days}</td>
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
                {station.data.pooled.paired_days} paired days in{" "}
                {station.data.years?.join(", ")}. Daily reporting windows
                differ; this limited check is descriptive and does not validate
                official warning prediction.
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
              href="/data/provider_benchmark.json"
              download
            >
              Original NASA benchmark <ArrowDown size={14} />
            </a>
          </div>
        </Panel>
      </section>
      <section id="original-study">
        <details className="study-disclosure">
          <summary>
            Original 26-input study · results, years and controlled experiments
          </summary>
          <Panel
            title="Original retrospective evaluation"
            subtitle="May 2019–2026 · 248 Nagpur days · 8 proxy-positive days"
            aside={
              <label className="field compact">
                Decision policy
                <select
                  value={policy}
                  onChange={(e) => setPolicy(e.target.value)}
                >
                  <option value="sensitive">Sensitivity · F2</option>
                  <option value="balanced">Balanced · F1</option>
                </select>
              </label>
            }
          >
            <ScoreTable metrics={oldMetrics} />
            <p className="fine-print">
              F2 prioritizes recall; F1 balances precision and recall. Both
              operating thresholds were chosen on May 2015–2018. These are
              retrospective results on previously examined years.
            </p>
          </Panel>
          <Panel title="Original chronological partitions">
            <div className="split-timeline">
              {model?.splits?.map((s, i) => (
                <div key={s.name}>
                  <span>0{i + 1}</span>
                  <strong>{s.years}</strong>
                  <p>{s.name}</p>
                  <small>
                    {s.days} days · {s.positives} positives
                  </small>
                </div>
              ))}
            </div>
            <p className="panel-copy">
              The scaler and reference normals fit training years only. March
              initializes lag history; observed inputs always stop at T−1.
              Missing or noncontiguous history fails explicitly.
            </p>
          </Panel>
          {experiments.data?.yearly_results && (
            <Panel title="Every evaluation May">
              <div className="table-scroll" tabIndex={0}>
                <table>
                  <thead>
                    <tr>
                      <th>Year</th>
                      <th>Positive days</th>
                      <th>Detected</th>
                      <th>Missed</th>
                      <th>False alarms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.data.yearly_results.map((r) => (
                      <tr key={r.year}>
                        <td>{r.year}</td>
                        <td>{r.positive_days}</td>
                        <td>{r.detected}</td>
                        <td>{r.missed}</td>
                        <td>{r.false_alarms}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          <Panel
            title="Controlled architecture and learning studies"
            subtitle="8/16/32 hidden neurons · two sigmoid activations · two losses · class-weight and learning-factor comparisons"
            aside={
              <label className="field compact">
                Activation
                <select
                  value={activation}
                  onChange={(e) => setActivation(e.target.value)}
                >
                  <option value="all">All activations</option>
                  <option value="binary_sigmoid">Binary sigmoid</option>
                  <option value="bipolar_sigmoid">Bipolar sigmoid</option>
                </select>
              </label>
            }
          >
            <div className="table-scroll study-table" tabIndex={0}>
              <table>
                <thead>
                  <tr>
                    <th>Activation</th>
                    <th>Hidden</th>
                    <th>Loss</th>
                    <th>Weight</th>
                    <th>η / μ</th>
                    <th>Seed</th>
                    <th>Selection AP</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r, i) => (
                    <tr key={i}>
                      <td>{r.activation?.replaceAll("_", " ")}</td>
                      <td>{r.hidden}</td>
                      <td>{r.loss?.replaceAll("_", " ")}</td>
                      <td>
                        {r.positive_weight_mode} ·{" "}
                        {number(r.positive_weight, 2)}
                      </td>
                      <td>
                        {r.learning_rate} / {r.momentum}
                      </td>
                      <td>{r.seed}</td>
                      <td>{number(r.pr_auc, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="panel-bottom">
              <span>{runs.length} recorded runs</span>
              <Download name="experiments" />
            </div>
          </Panel>
          <DataNote
            data={model}
            transport={resource.transport}
            error={resource.error}
          />
        </details>
      </section>
      <section id="lab-mapping">
        <Panel
          title="From the lab to the application"
          subtitle="The primary ANN uses Experiments 1, 6 and 7 directly. Perceptron and delta learning supply comparison models; the other examples explain the foundation."
        >
          <div className="table-scroll" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th>Experiment</th>
                  <th>Concept</th>
                  <th>Where it connects</th>
                  <th>Lab CO</th>
                </tr>
              </thead>
              <tbody>
                {syllabus.map((row) => (
                  <tr key={row[0]}>
                    {row.map((s, i) => (
                      <td key={i}>{s}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Link className="text-link" href="/learn">
            Explore activation functions and backpropagation{" "}
            <ArrowUpRight size={14} />
          </Link>
        </Panel>
      </section>
      <div className="method-footnote">
        <h3>What the prediction means</h3>
        <p>
          The declared proxy is Tmax ≥45°C, or Tmax ≥40°C with departure ≥4.5°C
          from a historical normal. It approximates grid conditions. Official
          IMD declarations have station, spatial and persistence requirements
          beyond this model.
        </p>
        <div className="panel-bottom">
          <a
            className="text-link"
            href="https://mausam.imd.gov.in/pdfs/heatcolduser/morning_heat_bulletin.pdf"
            target="_blank"
            rel="noreferrer"
          >
            IMD criteria <ArrowUpRight size={14} />
          </a>
          <a
            className="text-link"
            href="https://github.com/sohxmm/heatline"
            target="_blank"
            rel="noreferrer"
          >
            Source & reproducibility <ArrowUpRight size={14} />
          </a>
        </div>
      </div>
      <DataNote
        data={study.data ?? model}
        transport={study.data ? study.transport : resource.transport}
        error={study.data ? study.error : resource.error}
      />
    </div>
  );
}
