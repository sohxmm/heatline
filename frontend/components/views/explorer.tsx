"use client";
import { useState } from "react";
import { Info } from "lucide-react";
import {
  useResource,
  number,
  percent,
  dayLabel,
  type History,
  type Model,
} from "@/lib/data";
import {
  Tag,
  Empty,
  Heading,
  Panel,
  Metric,
  DataNote,
  Download,
} from "@/components/project-ui";
import { amber, cream, probabilitySeries } from "@/components/view-config";
import { Plot, Legend } from "@/components/charts";
import { DayReplay } from "@/components/day-replay";
import type { ReplayDetails } from "@/lib/data";

export function Explorer() {
  const resource = useResource<History>("history");
  const model = useResource<Model>("model");
  const [year, setYear] = useState(2026);
  const [day, setDay] = useState(1);
  const replay = useResource<ReplayDetails>(`replay/${year}`);
  const [errorsOnly, setErrorsOnly] = useState(false);
  const all = resource.data?.records ?? [];
  const years = Array.from(
    new Set(resource.data?.years ?? all.map((r) => Number(r.date.slice(0, 4)))),
  ).sort((a, b) => a - b);
  const rows = all.filter((r) => r.date.startsWith(String(year)));
  const errors = rows.filter((r) => r.actual !== r.prediction);
  const missed = rows.filter((r) => r.actual === 1 && r.prediction === 0);
  const threshold = model.data?.threshold ?? rows[0]?.threshold;
  const display = errorsOnly ? errors : rows;
  return (
    <div className="page">
      <Heading
        eyebrow="ASSESSED CORE / NAGPUR · MAY"
        title={
          <>
            Look back. <em>Learn more.</em>
          </>
        }
        description="Replay the ANN against historical reanalysis conditions. Every prediction uses earlier observations; outcomes are a declared grid-based heatwave proxy."
      >
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
      </Heading>
      <div className="view-context">
        <Tag tone="amber">Historical replay</Tag>
        <span>
          {year <= 2010
            ? "Training period"
            : year <= 2014
              ? "Architecture selection"
              : year <= 2018
                ? "Calibration / threshold selection"
                : "Retrospective evaluation"}
        </span>
        <span>31 May days · Nagpur</span>
      </div>
      <DataNote
        data={resource.data}
        transport={resource.transport}
        error={resource.error}
      />
      <div className="metric-strip">
        <Metric
          label="Observed proxy days"
          value={rows.length ? rows.filter((r) => r.actual === 1).length : "—"}
          detail={`May ${year}`}
        />
        <Metric
          label="Predicted positive days"
          value={
            rows.length ? rows.filter((r) => r.prediction === 1).length : "—"
          }
          detail={`Decision threshold ${number(threshold, 3)}`}
        />
        <Metric
          label="Missed proxy days"
          value={rows.length ? missed.length : "—"}
          detail="Actual positive · predicted negative"
        />
        <Metric
          label="All classification errors"
          value={rows.length ? errors.length : "—"}
          detail="False positives + false negatives"
        />
      </div>
      <DayReplay
        year={year}
        day={day}
        years={years}
        rows={rows}
        details={replay.data}
        loading={resource.loading || replay.loading}
        model={model.data}
        onYear={setYear}
        onDay={setDay}
      />
      <Panel
        title="Model output meets the historical record"
        subtitle="Platt-scaled output is an estimate; calibration is not guaranteed. Dashed reference shows the selected decision threshold."
        aside={<Tag>May {year}</Tag>}
      >
        <Legend series={probabilitySeries} />
        <Plot
          data={rows.map((r) => ({ ...r, probability: r.probability }))}
          x="date"
          series={probabilitySeries}
          xFormat={(v) => dayLabel(v, { day: "numeric", month: "short" })}
          domain={[0, 1]}
          reference={threshold}
          yFormat={(v) => `${Math.round(v * 100)}%`}
          height={300}
          area
        />
      </Panel>
      <Panel
        title="Temperature context"
        subtitle="Observed daily maximum and the 1981–2010 reference normal. Neither is a live forecast."
      >
        <Legend
          series={[
            { key: "tmax", name: "Daily maximum", color: amber },
            {
              key: "normal_tmax",
              name: "Training-period normal",
              color: cream,
              dashed: true,
            },
          ]}
        />
        <Plot
          data={rows.map((r) => ({ ...r }))}
          x="date"
          series={[
            { key: "tmax", name: "Daily maximum", color: amber },
            {
              key: "normal_tmax",
              name: "Training-period normal",
              color: cream,
              dashed: true,
            },
          ]}
          xFormat={(v) => dayLabel(v)}
          yFormat={(v) => `${v}°`}
          height={220}
        />
      </Panel>
      <Panel
        title="Inspect the individual days"
        subtitle="Errors stay visible. A model is useful only when its limitations are understood."
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
            <Download name="history" />
          </div>
        }
      >
        <div className="table-scroll history-table" tabIndex={0}>
          <table>
            <caption>
              Historical predictions and observed proxy outcomes for May {year}.
            </caption>
            <thead>
              <tr>
                <th>Date</th>
                <th>Maximum °C</th>
                <th>Raw ANN score</th>
                <th>Platt-scaled output</th>
                <th>Proxy outcome</th>
                <th>Prediction</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {display.map((r) => (
                <tr key={r.date}>
                  <td>
                    <a
                      className="text-link"
                      href="#day-replay"
                      onClick={() => setDay(Number(r.date.slice(8, 10)))}
                    >
                      {dayLabel(r.date)}
                    </a>
                  </td>
                  <td>{number(r.tmax)}</td>
                  <td>{number(r.score, 3)}</td>
                  <td>{percent(r.probability)}</td>
                  <td>{r.actual === 1 ? "Heatwave proxy" : "Negative"}</td>
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
        {!display.length && (
          <Empty>
            {resource.loading
              ? "Loading historical predictions…"
              : errorsOnly && rows.length
                ? "No classification errors in the selected month."
                : "No published predictions are available for this period."}
          </Empty>
        )}
      </Panel>
      <div className="notice">
        <Info size={17} />
        <p>
          The proxy approximates conditions on a reanalysis grid. It is not an
          official station-based IMD heatwave declaration. The 2019–2026 period
          is retrospective evaluation, not a newly unseen test.
        </p>
      </div>
    </div>
  );
}
