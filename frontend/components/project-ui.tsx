"use client";
import { motion, useReducedMotion } from "motion/react";
import { ArrowDown, CircleAlert, Database } from "lucide-react";
import { issueLabel, isStale, statusOf, type Envelope } from "@/lib/data";
export function Tag({
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
export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="empty">
      <CircleAlert size={20} />
      <p>{children}</p>
    </div>
  );
}
export function Heading({
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
export function Panel({
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
export function Metric({
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
export function DataNote({
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
export function Reveal({
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
export function Download({ name }: { name: string }) {
  return (
    <a className="text-link" href={`/data/${name}.json`} download>
      Download data <ArrowDown size={14} />
    </a>
  );
}
