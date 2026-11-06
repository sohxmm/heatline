"use client";
import Link from "next/link";
import { useState } from "react";
import { ArrowUpRight, CalendarDays, Network } from "lucide-react";
import {
  useResource,
  number,
  percent,
  dayLabel,
  type Outlook,
} from "@/lib/data";
import {
  DataNote,
  Heading,
  Metric,
  Panel,
  Empty,
} from "@/components/project-ui";
import { climateSeries } from "@/components/view-config";
import { Plot, Legend } from "@/components/charts";
import type { FullStudy } from "@/lib/study";

export function Landing() {
  const outlook = useResource<Outlook>("outlook");
  const study = useResource<FullStudy>("full_dataset_fit");
  const [cityId, setCity] = useState("nagpur");
  const [year, setYear] = useState(2026);
  const [day, setDay] = useState(20);
  const city = outlook.data?.cities?.find((c) => c.id === cityId);
  const days = city?.daily ?? [];
  const current = days[0];
  const source = study.data?.sources?.find((s) => s.id === "era5");
  const score = source?.metrics?.find((m) => m.model === "NumPy ANN");
  return (
    <div className="page dashboard-page">
      <Heading
        eyebrow="MAHARASHTRA · WEATHER & HEATWAVE HISTORY"
        title={
          <>
            Regional <em>dashboard.</em>
          </>
        }
        description="Seven-day weather forecasts and the Nagpur May prediction archive."
      >
        <Link className="button outlined" href="/explorer">
          May history <ArrowUpRight size={15} />
        </Link>
      </Heading>
      <section className="dashboard-weather">
        <div className="dashboard-weather-heading">
          <label className="field">
            Forecast city
            <select value={cityId} onChange={(e) => setCity(e.target.value)}>
              {(outlook.data?.cities?.length
                ? outlook.data.cities
                : [{ id: "nagpur", name: "Nagpur" }]
              ).map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          <div>
            <span className="muted">Provider forecast</span>
            <p>
              {current
                ? `${dayLabel(current.date)} – ${dayLabel(days[days.length - 1].date)}`
                : "Snapshot unavailable"}{" "}
              · IST
            </p>
          </div>
          <Link className="text-link" href={`/outlook?city=${cityId}`}>
            Hourly forecast <ArrowUpRight size={14} />
          </Link>
        </div>
        <div className="dashboard-weather-content">
          <div className="dashboard-temperature">
            <span>Daily maximum</span>
            <strong>
              {number(current?.tmax)}
              <small>°C</small>
            </strong>
            <div>
              <span>
                Overnight low<strong>{number(current?.tmin)}°</strong>
              </span>
              <span>
                Feels like max<strong>{number(current?.apparent_tmax)}°</strong>
              </span>
            </div>
          </div>
          <div className="dashboard-plot">
            <Legend series={climateSeries} />
            <Plot
              data={days.map((d) => ({ ...d }))}
              x="date"
              series={climateSeries}
              xFormat={(v) => dayLabel(v)}
              height={245}
              area
            />
          </div>
        </div>
        <DataNote
          data={outlook.data}
          transport={outlook.transport}
          error={outlook.error}
        />
      </section>
      <div
        className="regional-rail dashboard-cities"
        aria-label="Regional city forecasts"
      >
        {(outlook.data?.cities ?? []).map((c) => (
          <button
            key={c.id}
            type="button"
            aria-pressed={cityId === c.id}
            onClick={() => setCity(c.id)}
          >
            <span>{c.name}</span>
            <strong>
              {number(c.daily[0]?.tmax)}
              <small>°C</small>
            </strong>
            <ArrowUpRight size={14} />
          </button>
        ))}
      </div>
      <div className="dashboard-tools">
        <Panel
          title="Open a May day"
          subtitle="Prediction, recorded outcome and temperature context."
        >
          <div className="quick-history">
            <label className="field">
              Year
              <select
                aria-label="Year"
                value={year}
                onChange={(e) => setYear(Number(e.target.value))}
              >
                {Array.from({ length: 46 }, (_, i) => 2026 - i).map((y) => (
                  <option key={y}>{y}</option>
                ))}
              </select>
            </label>
            <label className="field">
              Day in May
              <select
                aria-label="Day in May"
                value={day}
                onChange={(e) => setDay(Number(e.target.value))}
              >
                {Array.from({ length: 31 }, (_, i) => i + 1).map((d) => (
                  <option key={d} value={d}>
                    {d} May
                  </option>
                ))}
              </select>
            </label>
            <Link
              className="button primary"
              href={`/explorer?year=${year}&day=${day}`}
            >
              <CalendarDays size={15} /> Open day
            </Link>
          </div>
        </Panel>
        <Panel
          title="Neural network"
          subtitle="Nagpur · May · NumPy backpropagation"
        >
          <div className="compact-network">
            <strong>
              {source?.architecture?.inputs ?? 26}
              <span>inputs</span>
            </strong>
            <span>→</span>
            <strong>
              {source?.architecture?.hidden ?? 16}
              <span>hidden</span>
            </strong>
            <span>→</span>
            <strong>
              1<span>output</span>
            </strong>
            <Link className="text-link" href="/learn">
              <Network size={15} /> Inspect network <ArrowUpRight size={14} />
            </Link>
          </div>
        </Panel>
      </div>
      <Panel
        title="Lab model fit"
        subtitle="ERA5 · May 1981–2026 · training-dataset results"
        aside={
          <Link className="text-link" href="/evidence">
            All results <ArrowUpRight size={14} />
          </Link>
        }
      >
        {score ? (
          <div className="metric-strip">
            <Metric
              label="Accuracy"
              value={percent(score.accuracy)}
              detail="All 1,426 fitted May dates"
            />
            <Metric
              label="Precision"
              value={percent(score.precision)}
              detail="Correct positive alerts"
            />
            <Metric
              label="Recall"
              value={percent(score.recall)}
              detail="Actual events detected"
            />
            <Metric
              label="F1 score"
              value={number(score.f1, 3)}
              detail={`${score.tp} detected · ${score.fp} false alarms`}
            />
          </div>
        ) : (
          <Empty>
            {study.loading
              ? "Loading model results…"
              : "Model results are unavailable in this snapshot."}
          </Empty>
        )}
      </Panel>
    </div>
  );
}
