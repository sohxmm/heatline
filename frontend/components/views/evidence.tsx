"use client";
import { useState } from "react";
import { Code2 as Github, ExternalLink } from "lucide-react";
import { useResource, number, type Model, type Experiments } from "@/lib/data";
import {
  Tag,
  Empty,
  Heading,
  Panel,
  DataNote,
  Download,
  Metric,
} from "@/components/project-ui";
import { amber } from "@/components/view-config";

const syllabus = [
  {
    exp: 1,
    title: "Activation functions",
    connection: "Identity, steps, sigmoids, ramp; controlled activation study",
    co: "CO1",
  },
  {
    exp: 2,
    title: "McCulloch–Pitts neuron",
    connection: "Fixed thresholds; NAND, NOR, and memory in the appendix",
    co: "CO1",
  },
  {
    exp: 3,
    title: "Hebbian learning",
    connection: "Binary and bipolar worked weight-association examples",
    co: "CO3",
  },
  {
    exp: 4,
    title: "Perceptron learning",
    connection: "Single-layer comparison; AND and XOR separability",
    co: "CO2",
  },
  {
    exp: 5,
    title: "Delta learning",
    connection: "Error-driven sigmoid updates; distinct from linear LMS",
    co: "CO2",
  },
  {
    exp: 6,
    title: "Multilayer perceptron",
    connection: "XOR hidden layer linked to the heatwave ANN",
    co: "CO2",
  },
  {
    exp: 7,
    title: "Error backpropagation",
    connection: "Numerical output/hidden errors and weight/bias updates",
    co: "CO2",
  },
];
export function Evidence() {
  const resource = useResource<Model>("model");
  const experiments = useResource<Experiments>("experiments");
  const model = resource.data;
  const metrics = model?.metrics ?? [];
  const [study, setStudy] = useState("all");
  const runs = (experiments.data?.runs ?? []).filter(
    (r) => study === "all" || r.activation === study,
  );
  const [policy, setPolicy] = useState("sensitive");
  const shownMetrics = metrics.map((m) =>
    policy === "balanced" && m.balanced ? { ...m, ...m.balanced } : m,
  );
  const previous = model?.previous_version?.metrics;
  const current = metrics[0];
  return (
    <div className="page">
      <Heading
        eyebrow="EVIDENCE / METHODOLOGY / LIMITATIONS"
        title={
          <>
            Show the work. <em>All of it.</em>
          </>
        }
        description="A reproducible experiment, a clearly bounded application, and honest results. The assessed contribution is the neural learning process—not a claim to replace official forecasts."
      >
        <Download name="model" />
      </Heading>
      <DataNote
        data={model}
        transport={resource.transport}
        error={resource.error}
      />
      <div className="scope-grid">
        <section>
          <Tag tone="amber">Assessed core</Tag>
          <h2>Nagpur · May</h2>
          <p>
            From-scratch NumPy backpropagation, activation studies, baselines,
            and chronological evaluation.
          </p>
        </section>
        <section>
          <Tag>Product extension</Tag>
          <h2>Six-city outlook</h2>
          <p>
            Provider weather forecasts, hourly profiles, historical context, and
            a useful regional interface.
          </p>
        </section>
        <section>
          <Tag>Optional theory</Tag>
          <h2>Beyond the lab</h2>
          <p>
            Fuzzy inference, genetic optimization, and the existing deeper
            PyTorch ensemble remain independent extensions.
          </p>
        </section>
      </div>
      {previous && current && (
        <Panel
          title="What changed, measured on the same May days"
          subtitle="Frozen ERA5 proxy labels and identical 2019–2026 dates. Previously inspected outcomes make this an exploratory retrospective comparison."
        >
          <div className="metric-strip">
            <Metric
              label="Proxy events detected"
              value={`${current.tp ?? 0} / ${(current.tp ?? 0) + (current.fn ?? 0)}`}
              detail={`Previous version: ${previous.tp} / ${(previous.tp ?? 0) + (previous.fn ?? 0)}`}
            />
            <Metric
              label="Recall"
              value={`${number((current.recall ?? 0) * 100, 1)}%`}
              detail={`Previous: ${number((previous.recall ?? 0) * 100, 1)}%`}
            />
            <Metric
              label="Average precision"
              value={number(current.pr_auc, 3)}
              detail={`Previous: ${number(previous.pr_auc, 3)}`}
            />
            <Metric
              label="False alarms"
              value={current.fp}
              detail={`Previous: ${previous.fp} · sensitivity costs alerts`}
            />
          </div>
          <p className="panel-copy">
            The revised study trains on April–June, compares positive-class
            weighting, and averages {model?.ensemble_size} small NumPy networks.
            The F2 operating point prioritizes recall; the balanced F1 policy is
            shown separately. Architecture selection uses May 2011–2014, and
            both thresholds use May 2015–2018.
          </p>
        </Panel>
      )}
      <Panel
        title="Retrospective evaluation"
        subtitle="2019–2026 · Nagpur May · identical target and evaluation dates for every baseline"
        aside={
          model?.decision_policy ? (
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
          ) : undefined
        }
      >
        <div className="table-scroll" tabIndex={0}>
          <table>
            <caption>
              Measured model comparisons. PR-AUC, precision, recall, F1, and
              Brier score.
            </caption>
            <thead>
              <tr>
                <th>Model</th>
                <th>PR-AUC ↑</th>
                <th>Precision ↑</th>
                <th>Recall ↑</th>
                <th>F1 ↑</th>
                <th>Brier ↓</th>
                <th>TN / FP / FN / TP</th>
              </tr>
            </thead>
            <tbody>
              {shownMetrics.map((m) => (
                <tr
                  key={m.model}
                  className={
                    m.model.toLowerCase().includes("ann") ? "highlight-row" : ""
                  }
                >
                  <td>{m.model}</td>
                  <td>{number(m.pr_auc, 3)}</td>
                  <td>{number(m.precision, 3)}</td>
                  <td>{number(m.recall, 3)}</td>
                  <td>{number(m.f1, 3)}</td>
                  <td>{number(m.brier, 3)}</td>
                  <td>
                    {[m.tn, m.fp, m.fn, m.tp]
                      .map((n) => number(n, 0))
                      .join(" / ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!metrics.length && (
          <Empty>
            {resource.loading
              ? "Loading measured results…"
              : "Results are not published yet. No performance numbers are estimated or invented."}
          </Empty>
        )}
        <p className="fine-print">
          PR-AUC is average precision. Brier score measures squared probability
          error; hard-output baselines are evaluated as 0/1 scores. Model
          selection and calibration precede this retrospective period. Simpler
          baselines may perform better.
        </p>
      </Panel>
      {experiments.data?.yearly_results && (
        <Panel
          title="Every evaluation May, including quiet years"
          subtitle="Detection and false alarms are reported together; zero-event years remain visible."
        >
          <div className="table-scroll" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Proxy-positive days</th>
                  <th>Detected</th>
                  <th>Missed</th>
                  <th>False alarms</th>
                </tr>
              </thead>
              <tbody>
                {experiments.data.yearly_results.map((row) => (
                  <tr key={row.year}>
                    <td>{row.year}</td>
                    <td>{row.positive_days}</td>
                    <td>{row.detected}</td>
                    <td>{row.missed}</td>
                    <td>{row.false_alarms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {experiments.data.onset && (
            <p className="fine-print">
              Event onsets: {experiments.data.onset.detected} detected of{" "}
              {experiments.data.onset.events}. An onset is a proxy-positive day
              after a negative day; continuation and onset are not the same
              task.
            </p>
          )}
          {experiments.data.uncertainty && (
            <p className="fine-print">
              May-year block bootstrap 95% interval: average precision{" "}
              {experiments.data.uncertainty.intervals.pr_auc
                .map((x) => number(x, 3))
                .join("–")}
              ; recall{" "}
              {experiments.data.uncertainty.intervals.recall
                .map((x) => `${number(x * 100, 1)}%`)
                .join("–")}
              . {experiments.data.uncertainty.limitation}
            </p>
          )}
        </Panel>
      )}
      <Panel
        title="Chronology is part of the method"
        subtitle="The future must not leak into a historical prediction."
      >
        <div className="split-timeline">
          {(
            model?.splits ?? [
              { name: "Training", years: "1981–2010", days: 0, positives: 0 },
              {
                name: "Architecture selection",
                years: "2011–2014",
                days: 0,
                positives: 0,
              },
              {
                name: "Calibration & threshold",
                years: "2015–2018",
                days: 0,
                positives: 0,
              },
              {
                name: "Retrospective evaluation",
                years: "2019–2026",
                days: 0,
                positives: 0,
              },
            ]
          ).map((s, i) => (
            <div key={s.name}>
              <span>0{i + 1}</span>
              <strong>{s.years}</strong>
              <p>{s.name}</p>
              {s.days > 0 && (
                <small>
                  {s.days} days · {s.positives} proxy positives
                </small>
              )}
            </div>
          ))}
        </div>
        <ul className="plain-list columns">
          <li>Normalization fits only the 1981–2010 training data.</li>
          <li>Reference normals use 1981–2010; later years are excluded.</li>
          <li>
            March observations initialize April–June training history; May
            inputs still end on T−1.
          </li>
          <li>
            Missing or noncontiguous history is rejected, never silently filled.
          </li>
          <li>2011–2014 selects architecture and objective.</li>
          <li>2015–2018 selects calibration and decision threshold.</li>
        </ul>
      </Panel>
      {experiments.data?.provider_benchmark && (
        <Panel
          title="Independent NASA POWER benchmark"
          subtitle="A separate data product and target population. These scores do not replace the ERA5 before/after comparison."
        >
          <p className="panel-copy">
            The NumPy network is also trained on NASA POWER meteorology with its
            own training-only reference normals. Evaluation covers{" "}
            {experiments.data.provider_benchmark.evaluation.days} May days in{" "}
            {experiments.data.provider_benchmark.evaluation.years}, including{" "}
            {experiments.data.provider_benchmark.evaluation.positive_days}{" "}
            proxy-positive days. ERA5 has 8 positive days over the same years,
            so these scores measure a different classification problem.
          </p>
          <div className="table-scroll" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th>NASA benchmark model</th>
                  <th>Average precision</th>
                  <th>F2-policy recall</th>
                  <th>F2-policy false alarms</th>
                  <th>F1-policy recall</th>
                  <th>F1-policy precision</th>
                  <th>F1-policy F1</th>
                  <th>F1-policy false alarms</th>
                </tr>
              </thead>
              <tbody>
                {experiments.data.provider_benchmark.metrics.map((m) => (
                  <tr key={m.model}>
                    <td>{m.model}</td>
                    <td>{number(m.pr_auc, 3)}</td>
                    <td>{number(m.recall, 3)}</td>
                    <td>{m.fp}</td>
                    <td>{number(m.balanced?.recall, 3)}</td>
                    <td>{number(m.balanced?.precision, 3)}</td>
                    <td>{number(m.balanced?.f1, 3)}</td>
                    <td>{m.balanced?.fp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fine-print">
            {experiments.data.provider_benchmark.time_standard}.{" "}
            {experiments.data.provider_benchmark.limitations.join(" ")}
          </p>
          <a
            className="text-link"
            href="/data/provider_benchmark.json"
            download
          >
            Download benchmark records and provenance
          </a>
        </Panel>
      )}
      <Panel
        title="Controlled experiments"
        subtitle="Repeated seeds, activation functions, hidden size, and learning settings."
        aside={
          <label className="field compact">
            Activation
            <select value={study} onChange={(e) => setStudy(e.target.value)}>
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
                <th>Positive weight</th>
                <th>Training months</th>
                <th>η</th>
                <th>μ</th>
                <th>Seed</th>
                <th>Selection PR-AUC</th>
                <th>Selection F1</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={i}>
                  <td>{r.activation?.replaceAll("_", " ")}</td>
                  <td>{r.hidden}</td>
                  <td>{r.loss?.replaceAll("_", " ")}</td>
                  <td>
                    {r.positive_weight_mode ?? "none"} ·{" "}
                    {number(r.positive_weight ?? 1, 2)}
                  </td>
                  <td>{r.training_window ?? "May"}</td>
                  <td>{number(r.learning_rate, 3)}</td>
                  <td>{number(r.momentum, 2)}</td>
                  <td>{r.seed}</td>
                  <td>{number(r.pr_auc, 3)}</td>
                  <td>{number(r.f1, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!runs.length && (
          <Empty>
            Controlled-study results have not been published for this selection.
          </Empty>
        )}
        <div className="panel-bottom">
          <span>{runs.length} measured runs</span>
          <Download name="experiments" />
        </div>
      </Panel>
      <div className="two-column">
        {experiments.data?.source_audit && (
          <Panel
            title="Weather sources do not define identical targets"
            subtitle="Training-era sensitivity check; providers were not selected using evaluation scores."
          >
            <p className="panel-copy">
              {experiments.data.source_audit.source_decision}
            </p>
            <p className="fine-print">
              ERA5 overlap: {experiments.data.source_audit.era5_overlap_days}{" "}
              days; maximum daily Tmax change{" "}
              {number(
                experiments.data.source_audit.max_daily_tmax_difference,
                4,
              )}
              °C; May label changes{" "}
              {experiments.data.source_audit.may_label_changes}.
            </p>
            <div className="table-scroll" tabIndex={0}>
              <table>
                <thead>
                  <tr>
                    <th>May year</th>
                    <th>NASA − ERA5 mean °C</th>
                    <th>NASA days ≥45°C</th>
                    <th>ERA5 days ≥45°C</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.data.source_audit.nasa_samples.map((row) => (
                    <tr key={row.year}>
                      <td>{row.year}</td>
                      <td>{number(row.mean_nasa_minus_era5, 2)}</td>
                      <td>{row.nasa_days_ge45}</td>
                      <td>{row.era5_days_ge45}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="fine-print">
              {experiments.data.source_audit.nasa_time_standard} NASA POWER is
              not substituted for ERA5 merely because it labels more days
              positive. See the downloadable source audit for station
              cross-checks and reporting-window limitations.
            </p>
            <a className="text-link" href="/data/source_audit.json" download>
              Download source audit
            </a>
          </Panel>
        )}
        <Panel
          title="What the labels mean"
          subtitle="A versioned proxy—not an official declaration."
        >
          <p className="panel-copy">
            The Nagpur plains proxy uses daily maximum ≥45°C, or maximum ≥40°C
            with departure from the reference normal ≥4.5°C. Thresholds are
            declared before fitting the model and are not adjusted to target a
            positive-class prevalence.
          </p>
          <p className="panel-copy">
            Gridded reanalysis approximates local conditions. Official IMD
            declarations depend on station observations and spatial/persistence
            requirements that this single-grid study does not reproduce.
          </p>
          <a
            className="text-link"
            href="https://mausam.imd.gov.in/pdfs/heatcolduser/morning_heat_bulletin.pdf"
            target="_blank"
            rel="noreferrer"
          >
            Read the IMD criteria <ExternalLink size={13} />
          </a>
        </Panel>
        <Panel
          title="What the model cannot claim"
          subtitle="Limitations belong alongside results."
        >
          <ul className="plain-list">
            {(
              model?.limitations ?? [
                "ERA5 reanalysis is delayed and cannot directly provide this model’s live inputs.",
                "Validation on one Nagpur grid does not establish validity for other cities.",
                "A calibrated model output can still be uncertain and miscalibrated under distribution shift.",
                "The previously inspected 2019–2026 period is retrospective evidence.",
              ]
            ).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <a
            className="text-link"
            href="https://open-meteo.com/en/docs/historical-weather-api"
            target="_blank"
            rel="noreferrer"
          >
            Data availability and provenance <ExternalLink size={13} />
          </a>
        </Panel>
      </div>
      <Panel
        title="The lab, connected to the project"
        subtitle="Experiment numbers and COs follow the supplied lab-printed mapping."
      >
        <div className="table-scroll" tabIndex={0}>
          <table>
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Lab concept</th>
                <th>Project connection</th>
                <th>Lab CO</th>
              </tr>
            </thead>
            <tbody>
              {syllabus.map((s) => (
                <tr key={s.exp}>
                  <td>0{s.exp}</td>
                  <td>{s.title}</td>
                  <td>{s.connection}</td>
                  <td>
                    <span className="co-tag">{s.co}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <section className="reproduce">
        <div>
          <p className="eyebrow">REPRODUCIBILITY, BUILT IN</p>
          <h2>
            From inputs to <em>evidence.</em>
          </h2>
          <p>
            Original outputs are preserved as legacy experiments. Corrected
            losses, independent gradient checks, seeded training, and versioned
            artifacts make the new results inspectable.
          </p>
        </div>
        <a
          className="button outlined"
          href="https://github.com/sohxmm/heatline"
          target="_blank"
          rel="noreferrer"
        >
          Explore the repository <Github size={17} />
        </a>
      </section>
    </div>
  );
}
