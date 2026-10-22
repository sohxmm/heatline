import type { Metadata } from "next";
import "./globals.css";
import { SiteShell } from "@/components/site-shell";
export const metadata: Metadata = {
  title: {
    default: "Heatline — understand the heat ahead",
    template: "%s · Heatline",
  },
  description:
    "A transparent neural network study of May heatwave conditions in Nagpur, with a six-city regional weather dashboard.",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>
        <SiteShell>{children}</SiteShell>
      </body>
    </html>
  );
}
