"use client";
import Link from "next/link";
import {
  ArrowUpRight,
  ArrowRight,
  MapPin,
  ArrowDown,
  Thermometer,
  CalendarDays,
  Network,
} from "lucide-react";
import {
  useResource,
  number,
  dayLabel,
  isStale,
  type Outlook,
  type History,
  type Model,
} from "@/lib/data";
import { Tag, DataNote, Reveal } from "@/components/project-ui";
import { amber, climateSeries } from "@/components/view-config";
import { Plot, Legend } from "@/components/charts";
import type { FullStudy } from "@/lib/study";

export function Landing() {
  const outlook = useResource<Outlook>("outlook");
  const history = useResource<History>("history");
  const model = useResource<Model>("model");
  const study = useResource<FullStudy>("full_dataset_study");
  const city = outlook.data?.cities?.find((c) => c.id === "nagpur");
  const rows = city?.daily ?? [];
  const replay = (history.data?.records ?? []).filter((r) =>
    r.date.startsWith("2026"),
  );
  const studySource = study.data?.sources?.find((s) => s.id === "era5");
  const studyScore = studySource?.metrics?.find((m) => m.model === "NumPy ANN");
  const score = studyScore ?? model.data?.metrics?.[0];
  const series = rows.length
    ? climateSeries
    : [{ key: "tmax", name: "Observed maximum", color: amber }];
  return (
    <>
      <section className="observatory-hero">
        <div className="ambient" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <Reveal className="hero-statement">
          <p className="eyebrow">
            <span className="live-dot" /> NAGPUR & THE MAHARASHTRA REGION
          </p>
          <h1>
            Understand the heat.
            <br />
            <em>See the evidence.</em>
          </h1>
          <p className="intro-copy">
            A regional weather observatory and a transparent neural network.
            Read the forecast, revisit a May day, or follow a prediction all the
            way back to its inputs.
          </p>
          <div className="hero-actions">
            <Link className="button primary" href="/explorer">
              Explore a May day <ArrowUpRight size={17} />
            </Link>
            <Link className="button outlined" href="/outlook">
              Regional weather <ArrowRight size={17} />
            </Link>
          </div>
        </Reveal>
        <div className="hero-index">
          <span>01 / WEATHER</span>
          <span>02 / HISTORY</span>
          <span>03 / NEURAL LEARNING</span>
          <a href="#regional-snapshot" aria-label="Scroll to regional snapshot">
            <ArrowDown size={20} />
          </a>
        </div>
      </section>
      <section className="observatory-board" id="regional-snapshot">
        <div className="board-sidebar">
          <p className="eyebrow">REGIONAL SNAPSHOT</p>
          <h2>
            <MapPin size={22} /> Nagpur
          </h2>
          <Tag
            tone={rows.length && !isStale(outlook.data) ? "green" : "neutral"}
          >
            {rows.length
              ? isStale(outlook.data)
                ? "Saved weather snapshot"
                : "Provider forecast"
              : "Historical replay"}
          </Tag>
          <div className="board-temperature">
            {number(rows[0]?.tmax ?? replay[0]?.tmax)}
            <span>°C</span>
          </div>
          <p>
            {rows[0]
              ? `${dayLabel(rows[0].date)} · daily maximum`
              : "May 2026 · observed maximum"}
          </p>
          <div className="board-mini">
            <div>
              <span>Overnight low</span>
              <strong>{number(rows[0]?.tmin)}°</strong>
            </div>
            <div>
              <span>Feels like max</span>
              <strong>{number(rows[0]?.apparent_tmax)}°</strong>
            </div>
          </div>
          <Link className="text-link" href="/outlook">
            View hourly conditions <ArrowUpRight size={15} />
          </Link>
        </div>
        <div className="board-chart">
          <div className="section-caption">
            <h3>{rows.length ? "The week in degrees" : "May in retrospect"}</h3>
            <span>°C · IST</span>
          </div>
          <Legend series={series} />
          <Plot
            data={(rows.length ? rows : replay).map((r) => ({ ...r }))}
            x="date"
            series={series}
            xFormat={(v) => dayLabel(v)}
            height={270}
            area
          />
          <DataNote
            data={rows.length ? outlook.data : history.data}
            transport={rows.length ? outlook.transport : history.transport}
            error={rows.length ? outlook.error : history.error}
          />
        </div>
      </section>
      <div className="regional-rail" aria-label="Regional city forecasts">
        {(outlook.data?.cities ?? []).map((c) => (
          <Link href={`/outlook?city=${c.id}`} key={c.id}>
            <span>{c.name}</span>
            <strong>
              {number(c.daily[0]?.tmax)}
              <small>°C</small>
            </strong>
            <ArrowUpRight size={14} />
          </Link>
        ))}
      </div>
      <section className="observatory-section">
        <div className="section-introduction">
          <p className="eyebrow">THREE QUESTIONS. REAL DATA.</p>
          <h2>
            Go beyond
            <br />
            <em>the temperature.</em>
          </h2>
          <p>
            Weather context, historical outcomes, and the learning process—each
            with a clear place to explore.
          </p>
        </div>
        <div className="editorial-links">
          {[
            {
              n: "01",
              icon: <Thermometer size={22} />,
              title: "What is happening across the region?",
              text: "Compare six cities, inspect the next seven days, and look inside an hourly profile.",
              href: "/outlook",
              label: "Open regional outlook",
            },
            {
              n: "02",
              icon: <CalendarDays size={22} />,
              title: "What did the model predict that day?",
              text: "Pick a May date, compare the prediction and recorded outcome, and inspect the earlier years behind the model.",
              href: "/explorer",
              label: "Open historical explorer",
            },
            {
              n: "03",
              icon: <Network size={22} />,
              title: "How does the neural network learn?",
              text: "Explore activation functions and a numerical backpropagation update, with the starting weights visible.",
              href: "/learn",
              label: "Follow the learning",
            },
          ].map((item) => (
            <Link key={item.n} href={item.href} className="editorial-link">
              <span className="editorial-number">{item.n}</span>
              <div>
                {item.icon}
                <h3>{item.title}</h3>
                <p>{item.text}</p>
                <span className="text-link">
                  {item.label} <ArrowRight size={14} />
                </span>
              </div>
              <ArrowUpRight size={20} />
            </Link>
          ))}
        </div>
      </section>
      <section className="research-summary">
        <div>
          <p className="eyebrow">THE ASSESSED STUDY</p>
          <h2>
            A small network.
            <br />
            <em>An inspectable result.</em>
          </h2>
          <p>
            {studyScore ? "52" : "26"} inputs → 16 hidden neurons → 1 output.
            Explicit NumPy backpropagation for May heatwave-proxy conditions in
            Nagpur.
          </p>
          <Link className="text-link" href="/evidence">
            Review results and methodology <ArrowUpRight size={15} />
          </Link>
        </div>
        <div className="research-numbers">
          <div>
            <strong>
              {score
                ? `${score.tp} / ${(score.tp ?? 0) + (score.fn ?? 0)}`
                : "—"}
            </strong>
            <span>proxy-positive days detected</span>
          </div>
          <div>
            <strong>{score?.fp ?? "—"}</strong>
            <span>false alarms</span>
          </div>
          <p>
            {studyScore
              ? "May 1996–2026 · ERA5 walk-forward evaluation · 961 held-out days."
              : "May 2019–2026 · ERA5 retrospective evaluation · sensitivity policy."}
            Event detection and false alarms belong together.
          </p>
        </div>
      </section>
    </>
  );
}
