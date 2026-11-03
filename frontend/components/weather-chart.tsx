"use client";
import { useMemo } from "react";
import { useReducedMotion } from "motion/react";
import { curveLinear } from "@visx/curve";
import { LineChart } from "./charts/line-chart";
import { Line } from "./charts/line";
import { Grid } from "./charts/grid";
import { XAxis } from "./charts/x-axis";
import { ChartTooltip } from "./charts/tooltip/chart-tooltip";
import { useYScale } from "./charts/chart-context";
import type { Day } from "@/lib/data";
import { number } from "@/lib/data";

function TemperatureAxis() {
  const scale = useYScale();
  return (
    <g aria-hidden="true">
      {scale.ticks(4).map((value) => (
        <text
          key={value}
          x={-9}
          y={scale(value)}
          textAnchor="end"
          dominantBaseline="middle"
          fill="#a0a397"
          fontSize={10}
        >
          {value}°
        </text>
      ))}
    </g>
  );
}

/** Bklit registry chart. Exact source numbers remain in the adjacent day controls/table. */
export function WeatherChart({ days }: { days: Day[] }) {
  const reduce = useReducedMotion();
  const data = useMemo(
    () =>
      days.map((d) => ({
        date: new Date(`${d.date}T12:00:00+05:30`),
        tmax: d.tmax,
        apparent_tmax: d.apparent_tmax,
      })),
    [days],
  );
  if (!data.length)
    return (
      <div className="empty-chart">No provider measurements are available.</div>
    );
  return (
    <div
      className="bklit-weather"
      role="img"
      aria-label="Bklit line chart of daily maximum and apparent maximum in Celsius. Exact values are in the day controls and regional table."
    >
      <LineChart
        data={data}
        style={{ height: 260 }}
        animationDuration={reduce ? 0 : 650}
        yDomainTween={!reduce}
        margin={{ top: 22, right: 16, bottom: 35, left: 42 }}
      >
        <Grid horizontal vertical={false} />
        <TemperatureAxis />
        <Line
          dataKey="tmax"
          stroke="#e7b46b"
          curve={curveLinear}
          animate={!reduce}
          fadeEdges={false}
        />
        <Line
          dataKey="apparent_tmax"
          stroke="#dcd7cb"
          curve={curveLinear}
          animate={!reduce}
          fadeEdges={false}
        />
        <XAxis numTicks={4} />
        <ChartTooltip
          damping={reduce ? 0 : 20}
          rows={(point) => [
            {
              label: "Maximum °C",
              value: number(point.tmax),
              color: "#e7b46b",
            },
            {
              label: "Apparent maximum °C",
              value: number(point.apparent_tmax),
              color: "#dcd7cb",
            },
          ]}
        />
      </LineChart>
    </div>
  );
}
