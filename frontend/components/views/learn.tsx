"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowUpRight, ArrowRight } from "lucide-react";
import { useResource, type Model, type Learning } from "@/lib/data";
import {
  Empty,
  Heading,
  Panel,
  DataNote,
  Download,
} from "@/components/project-ui";
import { amber, teal } from "@/components/view-config";
import { Plot, Legend } from "@/components/charts";
import { Button } from "@/components/ui/button";

export function Learn() {
  const model = useResource<Model>("model");
  const learning = useResource<Learning>("learning");
  const [activation, setActivation] = useState("binary_sigmoid");
  const [step, setStep] = useState(0);
  const activationData = useMemo(
    () =>
      Array.from({ length: 81 }, (_, i) => {
        const x = (i - 40) / 10;
        const y =
          activation === "binary_sigmoid"
            ? 1 / (1 + Math.exp(-x))
            : activation === "bipolar_sigmoid"
              ? 2 / (1 + Math.exp(-x)) - 1
              : activation === "identity"
                ? x
                : activation === "binary_step"
                  ? x >= 0
                    ? 1
                    : 0
                  : activation === "bipolar_step"
                    ? x >= 0
                      ? 1
                      : -1
                    : Math.max(0, Math.min(1, x));
        return { x, y };
      }),
    [activation],
  );
  const labels: Record<string, string> = {
    identity: "Identity",
    binary_step: "Binary step",
    bipolar_step: "Bipolar step",
    binary_sigmoid: "Binary sigmoid",
    bipolar_sigmoid: "Bipolar sigmoid",
    ramp: "Ramp",
  };
  const formulas: Record<string, string> = {
    identity: "f(z) = z",
    binary_step: "f(z) = 1 if z ≥ 0, otherwise 0",
    bipolar_step: "f(z) = +1 if z ≥ 0, otherwise −1",
    binary_sigmoid: "f(z) = 1 / (1 + exp(−z))",
    bipolar_sigmoid: "f(z) = 2 / (1 + exp(−z)) − 1",
    ramp: "f(z) = min(1, max(0, z))",
  };
  const trace = learning.data?.trace;
  const arch = model.data?.architecture;
  const curve =
    model.data?.training_history ?? learning.data?.training_history ?? [];
  return (
    <div className="page">
      <Heading
        eyebrow="FROM FIRST PRINCIPLES / LAB EXPERIMENTS 1–7"
        title={
          <>
            Small updates. <em>Real learning.</em>
          </>
        }
        description="Follow the same ideas from a single artificial neuron to the Nagpur network. The forward pass, errors, gradients, and bias updates are explicit NumPy operations."
      />
      <div className="learning-intro">
        <span>INPUTS</span>
        <ArrowRight size={16} />
        <span>ACTIVATION</span>
        <ArrowRight size={16} />
        <span>PREDICTION</span>
        <ArrowRight size={16} />
        <span>ERROR</span>
        <ArrowRight size={16} />
        <span>WEIGHT UPDATE</span>
      </div>
      <div className="two-column">
        <Panel
          title="01 / The artificial neuron"
          subtitle="Experiment 1 · CO1 · Activation functions"
          aside={
            <label className="sr-only" htmlFor="activation">
              Activation function
            </label>
          }
        >
          <div className="activation-control">
            <select
              id="activation"
              value={activation}
              onChange={(e) => setActivation(e.target.value)}
            >
              {Object.entries(labels).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
            <code>{formulas[activation]}</code>
          </div>
          <Plot
            data={activationData}
            x="x"
            series={[{ key: "y", name: labels[activation], color: amber }]}
            height={260}
          />
          <p className="panel-copy">
            A neuron forms a weighted sum plus a bias, then applies an
            activation. Binary and bipolar sigmoid are smooth, so their
            derivatives can propagate error through a hidden layer.
          </p>
        </Panel>
        <Panel
          title="02 / From XOR to climate inputs"
          subtitle="Experiment 6 · CO2 · Multilayer perceptron"
        >
          <div className="architecture">
            <div>
              <b>{arch?.inputs ?? 26}</b>
              <span>
                lagged climate
                <br />
                features
              </span>
            </div>
            <ArrowRight />
            <div className="hidden-layer">
              <b>{arch?.hidden ?? "8 / 16 / 32"}</b>
              <span>
                hidden neurons
                <br />
                {arch?.activation?.replaceAll("_", " ") ??
                  "selected by validation"}
              </span>
            </div>
            <ArrowRight />
            <div>
              <b>1</b>
              <span>
                sigmoid output
                <br />
                raw ANN score
              </span>
            </div>
          </div>
          <p className="panel-copy">
            A single perceptron cannot separate XOR with one straight boundary.
            A hidden layer can. Any benefit on weather data must still be
            demonstrated by measured comparisons.
          </p>
          <div className="formula">
            z = f(xV + v₀)
            <br />ŷ = σ(zW + w₀)
          </div>
          <p className="fine-print">
            Training loss:{" "}
            {arch?.loss?.replaceAll("_", " ") ??
              "half-squared-error reference and cross-entropy comparison"}
            . A sigmoid output alone does not establish calibration.
          </p>
        </Panel>
      </div>
      <Panel
        title="03 / One complete backpropagation step"
        subtitle="Experiment 7 · CO2 · Binary sigmoid · x = [0, 1], target = 1, η = 0.25, μ = 0"
      >
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
              <pre>
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
          <pre className="full-trace">
            {JSON.stringify(trace ?? { availability: "unavailable" }, null, 2)}
          </pre>
        </details>
      </Panel>
      <Panel
        title="04 / Learning over time"
        subtitle="Training and architecture-selection loss · the held-out evaluation years do not guide these updates."
        aside={<Download name="model" />}
      >
        <Legend
          series={[
            { key: "train_loss", name: "Training loss", color: amber },
            { key: "val_loss", name: "Selection loss", color: teal },
          ]}
        />
        <Plot
          data={curve.map((c) => ({ ...c }))}
          x="epoch"
          series={[
            { key: "train_loss", name: "Training loss", color: amber },
            { key: "val_loss", name: "Selection loss", color: teal },
          ]}
          height={280}
        />
      </Panel>
      <div className="two-column">
        <Panel title="One objective, one correct derivative">
          <div className="formula">
            E = ½(t − ŷ)²
            <br />
            ∂E/∂a = (ŷ − t)ŷ(1 − ŷ)
          </div>
          <p className="panel-copy">
            The textbook reference uses sigmoid with half-squared-error. The
            cross-entropy extension has a different output derivative: ŷ − t
            before any explicitly defined class weighting. Reporting one loss
            while optimizing another would make the learning account
            inconsistent.
          </p>
        </Panel>
        <Panel title="Learning rules have different roles">
          <ul className="plain-list">
            <li>
              <strong>Hebbian:</strong> associate co-active inputs and targets.
            </li>
            <li>
              <strong>Perceptron:</strong> correct classification mistakes at a
              hard threshold.
            </li>
            <li>
              <strong>Sigmoid delta:</strong> use the activation derivative for
              error-driven updates.
            </li>
            <li>
              <strong>ADALINE / LMS:</strong> optimize a linear output; distinct
              from sigmoid delta learning.
            </li>
          </ul>
          <Link className="text-link" href="/evidence">
            See the full syllabus mapping <ArrowUpRight size={14} />
          </Link>
        </Panel>
      </div>
      <DataNote
        data={learning.data}
        transport={learning.transport}
        error={learning.error}
      />
    </div>
  );
}
