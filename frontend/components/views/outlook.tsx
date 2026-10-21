"use client";
import { useState } from "react";
import {
  ArrowUpRight,
  MapPin,
  Layers3,
  Thermometer,
  Wind,
  Info,
} from "lucide-react";
import {
  useResource,
  number,
  percent,
  dayLabel,
  type Outlook,
  type City,
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
import {
  amber,
  cream,
  cityNames,
  climateSeries,
} from "@/components/view-config";
import { Plot, Legend } from "@/components/charts";
import { Button } from "@/components/ui/button";
import { WeatherChart } from "@/components/weather-chart";

export function Regional() {
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
