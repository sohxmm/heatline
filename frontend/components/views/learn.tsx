"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import {
  useResource,
  number,
  type Curve,
  type Envelope,
  type Learning,
} from "@/lib/data";
import {
  DataNote,
  Download,
  Empty,
  Heading,
  Metric,
  Panel,
  Tag,
} from "@/components/project-ui";
import { Plot, Legend } from "@/components/charts";
import { amber, teal } from "@/components/view-config";
import { Button } from "@/components/ui/button";

interface Member {
  seed: number;
  epochs: number;
  best_epoch?: number;
  training_history?: Curve[];
}
interface Fit {
  id?: string;
  years?: number[];
  fit_years?: number[];
  selection_years?: number[];
  calibration_years?: number[];
  evaluation_years?: number[];
  days?: number;
  positive_days?: number;
  positive_weight?: number;
  epochs?: number;
  members?: Member[];
  training_history?: Curve[];
  counts?: Record<string, { days: number; positives: number }>;
}
interface NetworkSource {
  id: string;
  label: string;
  feature_names?: string[];
  architecture: {
    inputs: number;
    hidden: number;
    output?: number;
    activation: string;
    loss: string;
    ensemble_size: number;
  };
  folds: Fit[];
  final_fit: Fit;
  initialization?: {
    method: string;
    generator: string;
    input_hidden_limit: number;
    hidden_output_limit: number;
    zero_biases: boolean;
    seeds: number[];
    members: {
      seed: number;
      first_input: string;
      first_input_weights: number[];
      output_weights: number[];
    }[];
  };
}
interface NetworkStudy extends Envelope {
  protocol?: {
    hyperparameters?: {
      learning_rate: number;
      momentum: number;
      batch_size: number;
      max_epochs: number;
      patience: number;
    };
  };
  sources?: NetworkSource[];
}
const range = (years?: number[]) => years?.join("–") ?? "Unavailable";
const lossSeries = [
  { key: "train_loss", name: "Training loss", color: amber },
  { key: "val_loss", name: "Stopping loss", color: teal },
];

export function Learn() {
  const [experiment, setExperiment] = useState("full");
  const study = useResource<NetworkStudy>(
    experiment === "full" ? "full_dataset_fit" : "full_dataset_study",
  );
  const [sourceId, setSource] = useState("era5");
  const [block, setBlock] = useState("final");
  const [seed, setSeed] = useState("42");
  const source = study.data?.sources?.find((s) => s.id === sourceId);
  const arch = source?.architecture;
  const fold = source?.folds.find((f) => f.id === block);
  const fit = block === "final" ? source?.final_fit : fold;
  const member = fit?.members?.find((m) => String(m.seed) === seed);
  const curves =
    seed === "mean" ? fit?.training_history : member?.training_history;
  const initial = source?.initialization;
  const initialMember = initial?.members.find((m) => String(m.seed) === seed);
  const factors = study.data?.protocol?.hyperparameters;
  const n = arch?.inputs,
    h = arch?.hidden,
    o = arch?.output ?? 1;
  const parameterCount =
    n !== undefined && h !== undefined ? n * h + h + h * o + o : undefined;
  const series = block === "final" ? lossSeries.slice(0, 1) : lossSeries;
  return (
    <div className="page network-workbench">
      <Heading
        eyebrow="03 / NETWORK WORKBENCH"
        title={
          <>
            See what the network <em>learned from.</em>
          </>
        }
        description="Choose a climate source, a training run and a member. Inspect its actual architecture, initial connections and recorded learning curve."
      >
        <Download
          name={
            experiment === "full" ? "full_dataset_fit" : "full_dataset_study"
          }
        />
      </Heading>
      <div className="inline-controls network-controls">
        <label className="field">
          Experiment
          <select
            aria-label="Experiment"
            value={experiment}
            onChange={(e) => {
              setExperiment(e.target.value);
              setBlock("final");
            }}
          >
            <option value="full">Lab · full dataset</option>
            <option value="rolling">
              Additional · chronological evaluation
            </option>
          </select>
        </label>
        <label className="field">
          Network source
          <select
            aria-label="Network source"
            value={sourceId}
            onChange={(e) => setSource(e.target.value)}
          >
            <option value="era5">ERA5 · primary</option>
            <option value="nasa_power">NASA POWER · comparison</option>
          </select>
        </label>
        <label className="field">
          Training run
          <select
            aria-label="Training run"
            value={block}
            onChange={(e) => setBlock(e.target.value)}
          >
            <option value="final">
              {experiment === "full"
                ? "All rows · 300 epochs"
                : "Full-data refit"}
            </option>
            {source?.folds.map((f) => (
              <option key={f.id} value={f.id}>
                Before evaluation {range(f.evaluation_years)}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Network member
          <select
            aria-label="Network member"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          >
            <option value="mean">Mean of all members</option>
            {(initial?.seeds ?? fit?.members?.map((m) => m.seed) ?? []).map(
              (s) => (
                <option key={s} value={s}>
                  Seed {s}
                </option>
              ),
            )}
          </select>
        </label>
      </div>
      <div className="view-context">
        <Tag tone="amber">
          {block === "final"
            ? "Full-dataset fit · in-sample"
            : "Earlier-years training run"}
        </Tag>
        <span>{source?.label ?? "Loading network metadata"}</span>
        <span>{study.data?.model_version}</span>
      </div>
      {source && arch ? (
        <>
          <div className="learning-facts" data-testid="network-settings">
            <div>
              <span>Update algorithm</span>
              <strong>Backpropagation · momentum SGD</strong>
            </div>
            <div>
              <span>Learning factors</span>
              <strong>
                η = {factors?.learning_rate ?? "—"} · μ ={" "}
                {factors?.momentum ?? "—"}
              </strong>
            </div>
            <div>
              <span>Batch / positive-class weight</span>
              <strong>
                {factors?.batch_size ?? "—"} / {number(fit?.positive_weight, 3)}
              </strong>
            </div>
            <div>
              <span>Fitting years</span>
              <strong>{range(fold?.fit_years ?? fit?.years)}</strong>
            </div>
          </div>
          <div className="two-column">
            <Panel
              title="The selected network"
              subtitle="Experiment 6 · multilayer perceptron · CO2"
            >
              <div className="architecture" data-testid="network-architecture">
                <div>
                  <b>{n}</b>
                  <span>climate inputs</span>
                </div>
                <ArrowRight />
                <div className="hidden-layer">
                  <b>{h}</b>
                  <span>{arch.activation.replaceAll("_", " ")}</span>
                </div>
                <ArrowRight />
                <div>
                  <b>{o}</b>
                  <span>sigmoid score</span>
                </div>
              </div>
              <p className="panel-copy">
                {parameterCount} learned weights and biases per member.{" "}
                {arch.ensemble_size} networks are trained separately; their
                scores are averaged before calibration.
              </p>
              <div className="formula">
                z = f(xV + v₀)
                <br />ŷ = σ(zW + w₀)
              </div>
              <details>
                <summary>Connection dimensions and input names</summary>
                <p className="panel-copy">
                  V: {n} × {h} · hidden biases: {h} · W: {h} × {o} · output
                  biases: {o}. Each input is standardized with the selected
                  run’s fitting-data statistics.
                </p>
                <p className="fine-print">
                  {source.feature_names?.join(" · ") ??
                    "Feature metadata unavailable."}
                </p>
              </details>
            </Panel>
            <Panel
              title="What is optimized"
              subtitle="Experiment 7 · error backpropagation · CO2"
            >
              <div className="formula">
                L = mean[½ q(t)(ŷ − t)²]
                <br />
                Δθ = −η ∇L + μ Δθprevious
              </div>
              <p className="panel-copy">
                Objective: {arch.loss.replaceAll("_", " ")}. Positive targets
                multiply both loss and gradient by{" "}
                {number(fit?.positive_weight, 3)}; negative targets use weight
                1. Backpropagation updates connection weights and biases.
              </p>
              <div className="metric-strip">
                <Metric
                  label="Fitting days"
                  value={fit?.counts?.fit?.days ?? fit?.days ?? "—"}
                />
                <Metric
                  label="Positive targets"
                  value={
                    fit?.counts?.fit?.positives ?? fit?.positive_days ?? "—"
                  }
                />
                <Metric
                  label={
                    seed === "mean" ? "Displayed epochs" : "Executed epochs"
                  }
                  value={
                    seed === "mean"
                      ? (curves?.length ?? "—")
                      : (member?.epochs ?? "—")
                  }
                />
              </div>
              <p className="fine-print">
                {fold
                  ? "Stopping " +
                    range(fold.selection_years) +
                    " · calibration " +
                    range(fold.calibration_years) +
                    " · evaluation " +
                    range(fold.evaluation_years) +
                    ". The evaluation days do not guide these updates."
                  : experiment === "full"
                    ? "All 4,186 rows fit preprocessing, weights, calibration and the decision threshold. Training runs for 300 fixed epochs without a validation partition. Its curve is training fit only."
                    : "This final refit uses every available seasonal training row. Its curve is training fit only; it has no independent validation-loss curve."}
              </p>
            </Panel>
          </div>
          <Panel
            title="Recorded learning curve"
            subtitle={
              seed === "mean"
                ? "Mean of the saved member histories over their common epoch span. Epoch numbers start at 1."
                : "Seed " +
                  seed +
                  " · actual saved epoch history · epochs count from 1" +
                  (member?.best_epoch
                    ? " · restored best epoch " + member.best_epoch
                    : "")
            }
          >
            {curves?.length ? (
              <>
                <Legend series={series} />
                <Plot
                  data={curves.map((c) => ({ ...c }))}
                  x="epoch"
                  series={series}
                  height={270}
                />
                <details>
                  <summary>Exact epoch values</summary>
                  <div className="table-scroll" tabIndex={0}>
                    <table data-testid="network-curve-values">
                      <caption>
                        {source.label} ·{" "}
                        {block === "final"
                          ? "full-data refit"
                          : "before " + block}{" "}
                        · {seed === "mean" ? "member mean" : "seed " + seed} ·
                        weighted objective
                      </caption>
                      <thead>
                        <tr>
                          <th>Epoch</th>
                          <th>Training loss</th>
                          {block !== "final" && <th>Stopping loss</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {curves.map((c) => (
                          <tr key={c.epoch}>
                            <td>{c.epoch}</td>
                            <td>{number(c.train_loss, 6)}</td>
                            {block !== "final" && (
                              <td>{number(c.val_loss, 6)}</td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              </>
            ) : (
              <Empty>
                No recorded curve is available for this run and member. No curve
                from another experiment is substituted.
              </Empty>
            )}
          </Panel>
          <Panel
            title="Starting connections"
            subtitle="Xavier initialization · Experiment 6/7"
          >
            {initial ? (
              <>
                <div className="learning-facts">
                  <div>
                    <span>Method</span>
                    <strong>{initial.method}</strong>
                  </div>
                  <div>
                    <span>Input → hidden range</span>
                    <strong>±{number(initial.input_hidden_limit, 6)}</strong>
                  </div>
                  <div>
                    <span>Hidden → output range</span>
                    <strong>±{number(initial.hidden_output_limit, 6)}</strong>
                  </div>
                  <div>
                    <span>Initial biases</span>
                    <strong>
                      {initial.zero_biases ? "All zero" : "See exported values"}
                    </strong>
                  </div>
                </div>
                <p className="fine-print">
                  {initial.generator}. Starting values are reconstructed from
                  the recorded seed and versioned initializer, rather than saved
                  initialization snapshots. They are separate from the learned
                  weights.
                </p>
                {initialMember ? (
                  <details>
                    <summary>Seed {seed} · inspect initial connections</summary>
                    <div className="table-scroll" tabIndex={0}>
                      <table data-testid="network-initial-values">
                        <caption>
                          Seed {seed} · {initialMember.first_input} to each
                          hidden neuron, and each hidden neuron to output
                        </caption>
                        <thead>
                          <tr>
                            <th>Hidden neuron</th>
                            <th>First input → hidden</th>
                            <th>Hidden → output</th>
                          </tr>
                        </thead>
                        <tbody>
                          {initialMember.first_input_weights.map(
                            (weight, i) => (
                              <tr key={i}>
                                <td>{i + 1}</td>
                                <td>{number(weight, 6)}</td>
                                <td>
                                  {number(initialMember.output_weights[i], 6)}
                                </td>
                              </tr>
                            ),
                          )}
                        </tbody>
                      </table>
                    </div>
                  </details>
                ) : (
                  <p className="panel-copy">
                    Choose a seeded member to inspect its starting connections.
                    The ensemble mean has no separate initialization.
                  </p>
                )}
              </>
            ) : (
              <Empty>
                Initialization metadata is unavailable for this source.
              </Empty>
            )}
          </Panel>
        </>
      ) : (
        <Empty>
          {study.loading
            ? "Loading the current network metadata…"
            : "Current network metadata is unavailable. The separate teaching exercises remain usable below."}
        </Empty>
      )}
      <div className="panel-bottom">
        <Link href="/explorer" className="text-link">
          Inspect a day’s actual forward calculation <ArrowUpRight size={14} />
        </Link>
        <Link href="/evidence" className="text-link">
          Compare model results <ArrowUpRight size={14} />
        </Link>
      </div>
      <DataNote
        data={study.data}
        transport={study.transport}
        error={study.error}
      />
      <ActivationLab />
      <TeachingTrace />
    </div>
  );
}

function ActivationLab() {
  const [activation, setActivation] = useState("binary_sigmoid");
  const labels: Record<string, string> = {
    identity: "Identity",
    binary_step: "Binary step",
    bipolar_step: "Bipolar step",
    binary_sigmoid: "Binary sigmoid",
    bipolar_sigmoid: "Bipolar sigmoid",
    ramp: "Ramp",
  };
  const data = useMemo(
    () =>
      Array.from({ length: 81 }, (_, i) => {
        const x = (i - 40) / 10;
        const y =
          activation === "binary_sigmoid"
            ? 1 / (1 + Math.exp(-x))
            : activation === "bipolar_sigmoid"
              ? Math.tanh(x / 2)
              : activation === "identity"
                ? x
                : activation === "binary_step"
                  ? Number(x >= 0)
                  : activation === "bipolar_step"
                    ? x >= 0
                      ? 1
                      : -1
                    : Math.max(0, Math.min(1, x));
        return { x, y };
      }),
    [activation],
  );
  return (
    <details className="study-disclosure">
      <summary>Lab exercise · activation functions</summary>
      <Panel
        title="Activation sandbox"
        subtitle="Experiment 1 · CO1 · changing this plot does not retrain the network above"
      >
        <label className="field">
          Activation function
          <select
            value={activation}
            onChange={(e) => setActivation(e.target.value)}
          >
            {Object.entries(labels).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <Plot
          data={data}
          x="x"
          series={[{ key: "y", name: labels[activation], color: amber }]}
          height={230}
        />
        <p className="fine-print">
          Binary sigmoid: σ(a), derivative y(1−y). Bipolar sigmoid: tanh(a/2),
          derivative (1−z²)/2. A hard step has no useful gradient for this
          backpropagation network.
        </p>
      </Panel>
    </details>
  );
}

function TeachingTrace() {
  const learning = useResource<Learning>("learning");
  const [step, setStep] = useState(0);
  const trace = learning.data?.trace;
  return (
    <details className="study-disclosure">
      <summary>Lab exercise · 2–2–1 backpropagation</summary>
      <Panel
        title="One numerical weight update"
        subtitle="Experiment 7 · CO2 · separate teaching fixture, not the trained weather network"
      >
        <p className="panel-copy">
          Two inputs, two hidden neurons, one output. Both layers use binary
          sigmoid; x = [0, 1], target = 1, η = 0.25 and μ = 0. These manually
          specified values demonstrate one update; they are not the weather
          network’s starting weights.
        </p>
        <div
          className="trace-tabs"
          role="tablist"
          aria-label="Backpropagation steps"
        >
          {[
            "Initial weights",
            "Forward pass",
            "Error signals",
            "Updated weights",
          ].map((name, i) => (
            <Button
              role="tab"
              aria-selected={step === i}
              aria-controls={`trace-panel-${i}`}
              id={`trace-tab-${i}`}
              tabIndex={step === i ? 0 : -1}
              onKeyDown={(e) => {
                if (e.key === "ArrowRight") {
                  e.preventDefault();
                  setStep((i + 1) % 4);
                  document.getElementById(`trace-tab-${(i + 1) % 4}`)?.focus();
                }
                if (e.key === "ArrowLeft") {
                  e.preventDefault();
                  setStep((i + 3) % 4);
                  document.getElementById(`trace-tab-${(i + 3) % 4}`)?.focus();
                }
              }}
              className={step === i ? "selected" : ""}
              onClick={() => setStep(i)}
              key={name}
            >
              <span>{i + 1}</span>
              {name}
            </Button>
          ))}
        </div>
        <div
          role="tabpanel"
          id={`trace-panel-${step}`}
          aria-labelledby={`trace-tab-${step}`}
          className="trace-body"
        >
          <div>
            <h3>
              {
                [
                  "Start with known numbers.",
                  "Turn inputs into an output.",
                  "Send the error backwards.",
                  "Take one learning step.",
                ][step]
              }
            </h3>
            <p>
              {
                [
                  "These illustrative initial weights are displayed so the entire calculation can be reproduced independently. The supplied α = 0.25 is the learning rate η; momentum μ is disabled.",
                  "Multiply each input by its weight, add the hidden bias, and apply sigmoid. Repeat from the hidden layer to the output.",
                  "For half-squared-error, the update signal is δₖ = (t − ŷ)ŷ(1 − ŷ). Each hidden error uses the original output weights and its own sigmoid derivative.",
                  "Update each connection by η times its error signal and its input. Biases receive the same signal without an input multiplier. All gradients are computed before any weight is changed.",
                ][step]
              }
            </p>
            <div className="formula">
              {
                [
                  "x = [0, 1] · t = 1\nη = 0.25 · μ = 0",
                  "zⱼ = σ(Σᵢ xᵢvᵢⱼ + v₀ⱼ)\nŷ = σ(Σⱼ zⱼwⱼ + w₀)",
                  "δₖ = (t − ŷ)ŷ(1 − ŷ)\nδⱼ = δₖwⱼzⱼ(1 − zⱼ)",
                  "wⱼ ← wⱼ + ηδₖzⱼ\nvᵢⱼ ← vᵢⱼ + ηδⱼxᵢ",
                ][step]
              }
            </div>
          </div>
          <div className="trace-values">
            <span className="eyebrow">NUMERICAL FIXTURE</span>
            {trace ? (
              <pre tabIndex={0} aria-label="Teaching step values">
                {JSON.stringify(
                  [trace.initial, trace.forward, trace.deltas, trace.updated][
                    step
                  ],
                  null,
                  2,
                )}
              </pre>
            ) : (
              <Empty>
                The numerical fixture will appear after the reproducible
                experiment export is generated.
              </Empty>
            )}
          </div>
        </div>
        <details>
          <summary>
            Inspect the complete numerical trace and initial weights
          </summary>
          <pre
            className="full-trace"
            tabIndex={0}
            aria-label="Complete teaching trace"
          >
            {JSON.stringify(trace ?? { availability: "unavailable" }, null, 2)}
          </pre>
        </details>
      </Panel>
    </details>
  );
}
