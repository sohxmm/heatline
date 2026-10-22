import type { NextConfig } from "next";
import path from "node:path";
const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  turbopack: { root: path.resolve(__dirname) },
};
export default config;
