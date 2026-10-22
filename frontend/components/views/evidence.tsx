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
      <Panel
        title="Retrospective evaluation"
        subtitle="2019–2026 · Nagpur May · identical target and evaluation dates for every baseline"
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
              {metrics.map((m) => (
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
          <li>April observations initialize lagged features for May.</li>
          <li>
            Missing or noncontiguous history is rejected, never silently filled.
          </li>
          <li>2011–2014 selects architecture and objective.</li>
          <li>2015–2018 selects calibration and decision threshold.</li>
        </ul>
      </Panel>
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
          href="https://github.com/sohxmm/heatwave-ANN"
          target="_blank"
          rel="noreferrer"
        >
          Explore the repository <Github size={17} />
        </a>
      </section>
    </div>
  );
}
