import type { Metadata } from "next";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";

import { SITE_TITLE } from "@/lib/siteMeta";

import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const jetBrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: SITE_TITLE,
  description: "Static, multi-language time & space complexity analysis: the exact cost expression, how it was derived, and how sure it is.",
};

// Runs before hydration so the first paint already lands on the right
// theme -- no light/dark flash. Falls back to the design's intended
// default ("dark", the primary "Console" experience) when there's no
// stored preference and the OS reports no preference either.
const THEME_INIT_SCRIPT = `(function(){try{var s=localStorage.getItem("polyo-theme");var t=(s==="light"||s==="dark")?s:(window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark");document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme="dark";}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${jetBrainsMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
