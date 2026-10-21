"use client";
import {
  ResponsiveContainer,
  LineChart,
  AreaChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { useId } from "react";
type Datum = Record<string, unknown>;
export type Series = {
  key: string;
  name: string;
  color: string;
  dashed?: boolean;
};
export function Plot({
  data,
  x,
  series,
  height = 260,
  domain,
  reference,
  area = false,
  xFormat,
  yFormat,
}: {
  data: Datum[];
  x: string;
  series: Series[];
  height?: number;
  domain?: [number | string, number | string];
  reference?: number;
  area?: boolean;
  xFormat?: (v: string) => string;
  yFormat?: (v: number) => string;
}) {
  const id = useId().replace(/:/g, "");
  if (!data.length)
    return (
      <div className="empty-chart" style={{ height }}>
        No measurements are available for this view.
      </div>
    );
  const Chart = area ? AreaChart : LineChart;
  return (
    <div
      className="plot"
      style={{ height }}
      role="img"
      aria-label={
        series.map((s) => s.name).join(" and ") +
        " chart. Exact values are available in the data table."
      }
    >
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        <Chart
          data={data}
          margin={{ top: 12, right: 12, left: -12, bottom: 8 }}
          accessibilityLayer
        >
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="0%"
                stopColor={series[0].color}
                stopOpacity={0.22}
              />
              <stop offset="95%" stopColor={series[0].color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            vertical={false}
            stroke="#2b2c2e"
            strokeDasharray="3 5"
          />
          <XAxis
            dataKey={x}
            axisLine={false}
            tickLine={false}
            minTickGap={28}
            tick={{ fill: "#93938d", fontSize: 11 }}
            tickFormatter={xFormat}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            width={48}
            tick={{ fill: "#93938d", fontSize: 11 }}
            domain={domain ?? ["auto", "auto"]}
            tickFormatter={yFormat}
          />
          <Tooltip
            contentStyle={{
              background: "#202123",
              border: "1px solid #454640",
              borderRadius: 10,
              color: "#eeeae1",
              fontSize: 12,
            }}
            labelStyle={{ color: "#aaa99f", marginBottom: 5 }}
            formatter={(value) =>
              typeof value === "number" ? Number(value.toFixed(4)) : value
            }
            labelFormatter={(v) => (xFormat ? xFormat(String(v)) : String(v))}
          />
          {reference !== undefined && (
            <ReferenceLine
              y={reference}
              stroke="#b9b8af"
              strokeDasharray="5 5"
            />
          )}
          {series.map((s, i) =>
            area && i === 0 ? (
              <Area
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.name}
                stroke={s.color}
                strokeWidth={2}
                fill={`url(#${id})`}
                isAnimationActive={false}
                connectNulls={false}
              />
            ) : (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.name}
                stroke={s.color}
                strokeWidth={2}
                strokeDasharray={s.dashed ? "4 5" : undefined}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
                isAnimationActive={false}
                connectNulls={false}
              />
            ),
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  );
}
export function Legend({ series }: { series: Series[] }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.key}>
          <i style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}
