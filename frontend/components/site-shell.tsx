"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { MotionConfig } from "motion/react";
import { ArrowUpRight, Code2 as Github, Menu, X } from "lucide-react";
import { Button } from "./ui/button";
const nav = [
  { id: "outlook", href: "/outlook", label: "Regional outlook" },
  { id: "explorer", href: "/explorer", label: "May explorer" },
  { id: "learn", href: "/learn", label: "How it learns" },
  { id: "evidence", href: "/evidence", label: "Evidence" },
];
export function SiteShell({ children }: { children: React.ReactNode }) {
  const view = usePathname().split("/")[1] || "home";
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
        <main id="main">{children}</main>
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
