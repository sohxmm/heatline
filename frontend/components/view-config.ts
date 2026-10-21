import type { Series } from "./charts";
export const amber = "#e7b46b",
  cream = "#dcd7cb",
  teal = "#8bb9ac";
export const cityNames = [
  "Nagpur",
  "Akola",
  "Amravati",
  "Chandrapur",
  "Pune",
  "Mumbai",
];
export const climateSeries: Series[] = [
  { key: "tmax", name: "Maximum temperature", color: amber },
  {
    key: "apparent_tmax",
    name: "Apparent temperature",
    color: cream,
    dashed: true,
  },
];
export const probabilitySeries: Series[] = [
  { key: "probability", name: "Platt-scaled model output", color: amber },
  { key: "actual", name: "Observed proxy label", color: teal, dashed: true },
];
