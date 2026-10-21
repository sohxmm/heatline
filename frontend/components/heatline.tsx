"use client";
import Link from "next/link";
import { Button } from "./ui/button";
import { WeatherChart } from "./weather-chart";
import { useMemo, useState } from "react";
import { motion, MotionConfig, useReducedMotion } from "motion/react";
import {
  ArrowUpRight,
  ArrowRight,
  Activity,
  MapPin,
  ArrowDown,
  Layers3,
  Thermometer,
  Wind,
  Info,
  Code2 as Github,
  Menu,
  X,
  Check,
  Database,
  CircleAlert,
  ExternalLink,
  RefreshCw,
} from "lucide-react";
import { Plot, Legend, type Series } from "./charts";
import {
  useResource,
  number,
  percent,
  dayLabel,
  issueLabel,
  isStale,
  statusOf,
  type Envelope,
  type Outlook,
  type History,
  type Model,
  type Experiments,
  type Learning,
  type City,
  type HistoryRow,
} from "@/lib/data";

type View = "home" | "outlook" | "explorer" | "learn" | "evidence";
const nav = [
  { id: "outlook", href: "/outlook", label: "Regional outlook" },
  { id: "explorer", href: "/explorer", label: "May explorer" },
  { id: "learn", href: "/learn", label: "How it learns" },
  { id: "evidence", href: "/evidence", label: "Evidence" },
];
const amber = "#e7b46b",
  cream = "#dcd7cb",
  teal = "#8bb9ac";
const cityNames = [
  "Nagpur",
  "Akola",
  "Amravati",
  "Chandrapur",
  "Pune",
  "Mumbai",
];
const climateSeries: Series[] = [
  { key: "tmax", name: "Maximum temperature", color: amber },
  {
    key: "apparent_tmax",
    name: "Apparent temperature",
    color: cream,
    dashed: true,
  },
];
const probabilitySeries: Series[] = [
  { key: "probability", name: "Platt-scaled model output", color: amber },
  { key: "actual", name: "Observed proxy label", color: teal, dashed: true },
];
function Tag({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return (
    <span className={`tag ${tone}`}>
      <i />
      {children}
    </span>
  );
}
function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="empty">
      <CircleAlert size={20} />
      <p>{children}</p>
    </div>
  );
}
function Heading({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: React.ReactNode;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {children}
    </div>
  );
}
function Panel({
  title,
  subtitle,
  children,
  aside,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  aside?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}
function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: React.ReactNode;
  detail?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}
function DataNote({
  data,
  transport,
  error,
}: {
  data?: Envelope | null;
  transport?: string;
  error?: string | null;
}) {
  const stale = isStale(data);
  return (
    <div className={`data-note ${stale ? "stale" : ""}`} role="status">
      <Database size={14} />
      <div>
        <span>
          {data?.source ?? "Data source pending"} ·{" "}
          {data?.mode ?? "unavailable"} ·{" "}
          {stale ? "Stale snapshot" : statusOf(data?.availability)}
          {transport === "snapshot" ? " · saved snapshot" : ""}
        </span>
        <small>
          Issued {issueLabel(data?.issued_at)}
          {data?.model_version ? ` · ${data.model_version}` : ""}
          {data?.target_date ? ` · Target ${data.target_date}` : ""}
        </small>
        {error && <small>{error}</small>}
      </div>
    </div>
  );
}
function Reveal({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={false}
      whileInView={
        reduce ? { opacity: 1, y: 0 } : { opacity: [0.65, 1], y: [16, 0] }
      }
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: reduce ? 0 : 0.65, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}
function Download({ name }: { name: string }) {
  return (
    <a className="text-link" href={`/data/${name}.json`} download>
      Download data <ArrowDown size={14} />
    </a>
  );
}

export function Heatline({ view }: { view: View }) {
  const [menu, setMenu] = useState(false);
  return (
    <MotionConfig reducedMotion="user">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <div className="site-shell">
        <header className="site-header">
          <Link href="/" className="brand" aria-label="Heatline home">
            <span className="brand-mark">
              <i />
              <i />
              <i />
            </span>
            heatline<span className="brand-dot">.</span>
          </Link>
          <nav
            className={menu ? "nav open" : "nav"}
            id="mobile-nav"
            aria-label="Main navigation"
          >
            {nav.map((item) => (
              <Link
                onClick={() => setMenu(false)}
                key={item.id}
                href={item.href}
                aria-current={view === item.id ? "page" : undefined}
                className={view === item.id ? "active" : ""}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <a
            className="github-link"
            href="https://github.com/sohxmm/heatwave-ANN"
            target="_blank"
            rel="noreferrer"
            aria-label="View source on GitHub"
          >
            <Github size={18} />
            <span>Open source</span>
            <ArrowUpRight size={14} />
          </a>
          <Button
            className="mobile-toggle"
            onClick={() => setMenu(!menu)}
            aria-expanded={menu}
            aria-controls="mobile-nav"
            aria-label={menu ? "Close navigation" : "Open navigation"}
          >
            {menu ? <X /> : <Menu />}
          </Button>
        </header>
        <main id="main">
          {view === "home" ? (
            <Landing />
          ) : view === "outlook" ? (
            <Regional />
          ) : view === "explorer" ? (
            <Explorer />
          ) : view === "learn" ? (
            <Learn />
          ) : (
            <Evidence />
          )}
        </main>
        <footer className="site-footer">
          <Link href="/" className="brand">
            heatline<span className="brand-dot">.</span>
          </Link>
          <p>A neural learning study. A clearer climate story.</p>
          <div>
            <span>°C · Asia/Kolkata</span>
            <Link href="/evidence">
              Methodology <ArrowUpRight size={12} />
            </Link>
          </div>
        </footer>
      </div>
    </MotionConfig>
  );
}

function Landing() {
  const outlook = useResource<Outlook>("outlook");
  const history = useResource<History>("history");
  const city = outlook.data?.cities?.find((c) => c.id === "nagpur");
  const rows = city?.daily ?? [];
  const replay = (history.data?.records ?? []).filter((r) =>
    r.date.startsWith("2026"),
  );
  return (
    <>
      <section className="hero">
        <div className="ambient" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <Reveal>
          <div className="hero-kicker">
            <span className="live-dot" />
            CLIMATE INTELLIGENCE, WITH CONTEXT <span>01 / NAGPUR</span>
          </div>
          <h1>
            A clearer view of
            <br />
            the <em>heat ahead.</em>
          </h1>
          <p className="hero-description">
            From the weather outside to the neural network underneath. Explore
            regional conditions, revisit May heatwaves, and see exactly how an
            ANN learns.
          </p>
          <div className="hero-actions">
            <Link className="button primary" href="/outlook">
              Explore the outlook <ArrowUpRight size={17} />
            </Link>
            <Link className="button subtle" href="/learn">
              Inside the neural network <ArrowRight size={16} />
            </Link>
          </div>
          <div className="hero-meta">
            <span>
              <MapPin size={13} /> Maharashtra, India
            </span>
            <span>6 cities</span>
            <span>46 years of climate history</span>
          </div>
        </Reveal>
        <Reveal className="hero-preview">
          <div className="preview-heading">
            <span>
              <MapPin size={15} /> Nagpur
            </span>
            <Tag
              tone={rows.length && !isStale(outlook.data) ? "green" : "neutral"}
            >
              {rows.length ? "Provider forecast" : "Historical replay"}
            </Tag>
          </div>
          <div className="preview-title">
            <div>
              <span>
                {rows.length
                  ? "The next seven days"
                  : "May 2026 · retrospective"}
              </span>
              <h2>
                {rows.length ? (
                  <>
                    A week in <em>degrees.</em>
                  </>
                ) : (
                  <>
                    Every day tells <em>a story.</em>
                  </>
                )}
              </h2>
            </div>
            <span className="temperature">
              {number(rows[0]?.tmax ?? replay[0]?.tmax)}
              <sup>°C</sup>
            </span>
          </div>
          <Legend
            series={
              rows.length
                ? climateSeries
                : [
                    {
                      key: "tmax",
                      name: "Daily maximum temperature",
                      color: amber,
                    },
                  ]
            }
          />
          <Plot
            data={(rows.length ? rows : replay).map((r) => ({ ...r }))}
            x="date"
            series={
              rows.length
                ? climateSeries
                : [
                    {
                      key: "tmax",
                      name: "Daily maximum temperature",
                      color: amber,
                    },
                  ]
            }
            xFormat={(v) => dayLabel(v)}
            height={245}
            area
          />
          <DataNote
            data={rows.length ? outlook.data : history.data}
            transport={rows.length ? outlook.transport : history.transport}
            error={rows.length ? outlook.error : history.error}
          />
          <Link
            href={rows.length ? "/outlook" : "/explorer"}
            className="preview-link"
          >
            Open the full {rows.length ? "outlook" : "replay"}
            <ArrowUpRight size={17} />
          </Link>
        </Reveal>
      </section>
      <section className="scope-strip">
        <div>
          <span>THE ASSESSED CORE</span>
          <strong>Nagpur. May. Backpropagation.</strong>
        </div>
        <p>
          A NumPy ANN built from first principles, evaluated against simpler
          models, with every assumption in view.
        </p>
        <Link href="/evidence">
          <ArrowUpRight size={22} />
          <span className="sr-only">Read assessed scope</span>
        </Link>
      </section>
      <section className="home-section">
        <Reveal>
          <p className="eyebrow">THREE WAYS TO LOOK CLOSER</p>
          <h2>
            Useful on the surface.
            <br />
            <em>Transparent underneath.</em>
          </h2>
        </Reveal>
        <div className="feature-grid">
          {[
            {
              n: "01",
              icon: <Wind />,
              title: "Read the region",
              text: "Temperature, apparent temperature, and hourly profiles across six Maharashtra cities. Source and freshness always visible.",
              href: "/outlook",
            },
            {
              n: "02",
              icon: <Activity />,
              title: "Replay a heatwave",
              text: "Compare historical predictions with proxy outcomes in Nagpur. Find the missed events as well as the successful predictions.",
              href: "/explorer",
            },
            {
              n: "03",
              icon: <Layers3 />,
              title: "Follow the learning",
              text: "Trace one example from inputs to error, through gradients and weight updates. The mathematics stays in plain sight.",
              href: "/learn",
            },
          ].map((item) => (
            <Reveal key={item.n}>
              <Link href={item.href} className="feature">
                <span className="feature-number">
                  {item.n}
                  <ArrowUpRight size={18} />
                </span>
                {item.icon}
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </Link>
            </Reveal>
          ))}
        </div>
      </section>
      <section className="home-bottom">
        <div>
          <Tag>Research, with limits</Tag>
          <h2>
            Evidence before <em>confidence.</em>
          </h2>
          <p>
            Weather forecasts come from Open-Meteo. The assessed ANN is
            evaluated on historical Nagpur May data. These are two distinct
            parts of the project.
          </p>
        </div>
        <Link className="button outlined" href="/evidence">
          Read the methodology <ArrowUpRight size={16} />
        </Link>
      </section>
    </>
  );
}

function Regional() {
  const resource = useResource<Outlook>("outlook");
  const [selected, setSelected] = useState("nagpur");
  const [day, setDay] = useState(0);
  const city = resource.data?.cities?.find((c) => c.id === selected);
  const days = city?.daily ?? [];
  const current = days[day];
  const hours = (city?.hourly ?? []).filter((h) =>
    current ? h.time.startsWith(current.date) : false,
  );
  const season = current
    ? [4, 5, 6].includes(Number(current.date.slice(5, 7)))
    : false;
  return (
    <div className="page">
      <Heading
        eyebrow="PRODUCT EXTENSION / REGIONAL WEATHER"
        title={
          <>
            The regional <em>outlook.</em>
          </>
        }
        description="Six cities, one clear view. Provider forecasts offer current weather context; the assessed ANN remains a historical Nagpur May study."
      >
        <Tag tone="amber">Weather-provider forecast</Tag>
      </Heading>
      <div className="city-selector" role="group" aria-label="Select a city">
        {cityNames.map((name) => (
          <Button
            key={name}
            className={selected === name.toLowerCase() ? "selected" : ""}
            onClick={() => {
              setSelected(name.toLowerCase());
              setDay(0);
            }}
          >
            <MapPin size={14} />
            {name}
          </Button>
        ))}
      </div>
      <DataNote
        data={city ?? resource.data}
        transport={resource.transport}
        error={resource.error}
      />
      {!season && current && (
        <div className="notice">
          <Info size={17} />
          <p>
            Outside the April–June study season. The weather outlook remains
            available; seasonal ANN coverage is unchanged.
          </p>
        </div>
      )}
      <div className="metric-strip">
        <Metric
          label={`${city?.name ?? cityNames.find((n) => n.toLowerCase() === selected)} · ${current ? dayLabel(current.date) : "Forecast"}`}
          value={
            <>
              {number(current?.tmax)}
              <span>°C</span>
            </>
          }
          detail="Daily maximum"
        />
        <Metric
          label="Feels like"
          value={
            <>
              {number(current?.apparent_tmax)}
              <span>°C</span>
            </>
          }
          detail="Apparent maximum · separate indicator"
        />
        <Metric
          label="Overnight low"
          value={
            <>
              {number(current?.tmin)}
              <span>°C</span>
            </>
          }
          detail="Daily minimum"
        />
        <Metric
          label="ANN prediction"
          value={<span className="small-value">Unavailable</span>}
          detail="Timely inputs require separate validation"
        />
      </div>
      <div className="two-column">
        <Panel
          title="Temperature through the week"
          subtitle="Choose a day below to inspect its hourly profile."
          aside={<Thermometer size={18} className="amber-icon" />}
        >
          <Legend series={climateSeries} />
          <WeatherChart days={days} />
          <div className="day-selector">
            {days.map((d, i) => (
              <Button
                key={d.date}
                className={i === day ? "selected" : ""}
                onClick={() => setDay(i)}
              >
                <span>{dayLabel(d.date, { weekday: "short" })}</span>
                <strong>{number(d.tmax, 0)}°</strong>
                <small>{dayLabel(d.date)}</small>
              </Button>
            ))}
          </div>
        </Panel>
        <Panel
          title="Context matters"
          subtitle="Three signals with different meanings."
        >
          <div className="context-item">
            <div className="context-icon">
              <Thermometer size={18} />
            </div>
            <div>
              <h3>High temperature screening</h3>
              <p>
                {current?.temperature_indicator?.exceeds_threshold
                  ? `Maximum meets the ${current.temperature_indicator.threshold_c}°C screening threshold.`
                  : current?.temperature_indicator?.threshold_c
                    ? `Maximum is below the ${current.temperature_indicator.threshold_c}°C screening threshold.`
                    : "Temperature screening is not available for this snapshot."}
              </p>
              <small>
                A temperature threshold alone is not a heatwave declaration.
              </small>
            </div>
          </div>
          <div className="context-item">
            <div className="context-icon">
              <Wind size={18} />
            </div>
            <div>
              <h3>Apparent temperature</h3>
              <p>
                A separate measure of how hot conditions may feel, combining
                meteorological influences.
              </p>
              <small>Not used as an IMD heatwave label.</small>
            </div>
          </div>
          <div className="context-item">
            <div className="context-icon">
              <Layers3 size={18} />
            </div>
            <div>
              <h3>Regional ANN · unavailable</h3>
              <p>
                {city?.ann?.reason ??
                  "The Nagpur model has not been validated for live provider inputs or these other cities."}
              </p>
            </div>
          </div>
        </Panel>
      </div>
      <Panel
        title="Hourly profile"
        subtitle={
          current
            ? `${city?.name} · ${dayLabel(current.date)} · local time (IST)`
            : "Select an available forecast day"
        }
      >
        <Legend
          series={[
            { key: "temperature", name: "Temperature", color: amber },
            {
              key: "apparent_temperature",
              name: "Apparent temperature",
              color: cream,
              dashed: true,
            },
          ]}
        />
        <Plot
          data={hours.map((h) => ({ ...h }))}
          x="time"
          series={[
            { key: "temperature", name: "Temperature", color: amber },
            {
              key: "apparent_temperature",
              name: "Apparent temperature",
              color: cream,
              dashed: true,
            },
          ]}
          xFormat={(v) => v.slice(11, 16)}
          height={280}
        />
        <details>
          <summary>View hourly measurements</summary>
          <div className="table-scroll" tabIndex={0}>
            <table>
              <caption>
                Hourly provider measurements in {city?.name}, Celsius and
                relative humidity percent.
              </caption>
              <thead>
                <tr>
                  <th>Time (IST)</th>
                  <th>Temperature °C</th>
                  <th>Apparent °C</th>
                  <th>Humidity %</th>
                </tr>
              </thead>
              <tbody>
                {hours.map((h) => (
                  <tr key={h.time}>
                    <td>{h.time.slice(11, 16)}</td>
                    <td>{number(h.temperature)}</td>
                    <td>{number(h.apparent_temperature)}</td>
                    <td>{number(h.humidity, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </Panel>
      <Panel
        title="Across the region"
        subtitle="Daily values for the selected date. Forecasts, not ANN predictions."
        aside={<Download name="outlook" />}
      >
        <div className="table-scroll" tabIndex={0}>
          <table>
            <thead>
              <tr>
                <th>City</th>
                <th>Target date</th>
                <th>Maximum °C</th>
                <th>Minimum °C</th>
                <th>Apparent max °C</th>
                <th>Heatwave declaration</th>
              </tr>
            </thead>
            <tbody>
              {(resource.data?.cities ?? []).map((c) => {
                const d = c.daily?.find((d) => d.date === current?.date);
                return (
                  <tr key={c.id}>
                    <td>
                      <Button
                        className="table-city"
                        onClick={() => {
                          setSelected(c.id);
                          setDay(0);
                        }}
                      >
                        {c.name}
                        <ArrowUpRight size={12} />
                      </Button>
                    </td>
                    <td>{d ? dayLabel(d.date) : "Unavailable"}</td>
                    <td>{number(d?.tmax)}</td>
                    <td>{number(d?.tmin)}</td>
                    <td>{number(d?.apparent_tmax)}</td>
                    <td>
                      <span className="muted">Unavailable</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!resource.data?.cities?.length && (
          <Empty>
            {resource.loading
              ? "Loading the weather snapshot…"
              : "No weather snapshot is available. The dashboard will populate when a valid provider snapshot is published."}
          </Empty>
        )}
      </Panel>
    </div>
  );
}

function Explorer() {
  const resource = useResource<History>("history");
  const model = useResource<Model>("model");
  const [year, setYear] = useState(2026);
  const [errorsOnly, setErrorsOnly] = useState(false);
  const all = resource.data?.records ?? [];
  const years =
    resource.data?.years ??
    Array.from(new Set(all.map((r) => Number(r.date.slice(0, 4))))).sort(
      (a, b) => b - a,
    );
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
        <div className="table-scroll" tabIndex={0}>
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
                  <td>{dayLabel(r.date)}</td>
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

function Learn() {
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
function Evidence() {
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
