import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "ExoReliability Lab",
  description: "Reliability of ML exoplanet-transit detection on Kepler light curves",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header>
          <Link href="/">ExoReliability Lab</Link>
          <span className="muted"> — Kepler Q1–Q17 DR25 pipeline validation (Milestone 1)</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
