"use client";
import Link from "next/link";
import {
  ArrowUpRight,
  ArrowRight,
  Activity,
  MapPin,
  Layers3,
  Wind,
} from "lucide-react";
import {
  useResource,
  number,
  dayLabel,
  isStale,
  type Outlook,
  type History,
} from "@/lib/data";
import { Tag, DataNote, Reveal } from "@/components/project-ui";
import { amber, climateSeries } from "@/components/view-config";
import { Plot, Legend } from "@/components/charts";

export function Landing() {
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
