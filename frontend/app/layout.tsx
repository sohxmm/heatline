import type { Metadata } from "next";
import "./globals.css";
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
      <body>{children}</body>
    </html>
  );
}
