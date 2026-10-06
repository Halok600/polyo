import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SITE_TITLE, tabTitle, useFaviconSpinner, useTabStatus } from "@/lib/tabStatus";

describe("tabTitle", () => {
  it("is the full site title when nothing is happening", () => {
    expect(tabTitle({ analysing: false, answer: null })).toBe(SITE_TITLE);
  });

  it("says it is analysing while a request is running, even if an old answer is still shown", () => {
    expect(tabTitle({ analysing: true, answer: null })).toBe("Analysing… · PolyO");
    expect(tabTitle({ analysing: true, answer: "O(n)" })).toBe("Analysing… · PolyO");
  });

  it("leads with the answer once there is one, so it can be read from another tab", () => {
    expect(tabTitle({ analysing: false, answer: "O(n * m)" })).toBe("O(n * m) · PolyO");
  });

  it("falls back to the site title for an empty answer instead of showing ' · PolyO'", () => {
    expect(tabTitle({ analysing: false, answer: "" })).toBe(SITE_TITLE);
  });

  it("is a real, descriptive title, not the bare product name", () => {
    expect(SITE_TITLE).toMatch(/PolyO/);
    expect(SITE_TITLE.length).toBeGreaterThan("PolyO".length + 10);
  });
});

function stubMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: reduced,
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

function addIconLinks() {
  document.head.innerHTML = `
    <link rel="icon" href="/favicon.ico" sizes="48x48">
    <link rel="icon" href="/icon.svg?x" type="image/svg+xml" sizes="any">
    <link rel="apple-touch-icon" href="/apple-icon.png" sizes="180x180">`;
  return [...document.querySelectorAll<HTMLLinkElement>('link[rel="icon"]')];
}

describe("useTabStatus", () => {
  beforeEach(() => {
    stubMotion(true); // no spinner here: only the title is under test
    document.head.innerHTML = "";
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sets the document title from the state, and follows it as it changes", () => {
    const { rerender } = renderHook((props) => useTabStatus(props), { initialProps: { analysing: false, answer: null as string | null } });
    expect(document.title).toBe(SITE_TITLE);
    rerender({ analysing: true, answer: null });
    expect(document.title).toBe("Analysing… · PolyO");
    rerender({ analysing: false, answer: "O(n^2)" });
    expect(document.title).toBe("O(n^2) · PolyO");
    rerender({ analysing: true, answer: "O(n^2)" }); // a second run: back to analysing
    expect(document.title).toBe("Analysing… · PolyO");
  });

  it("puts the plain title back when the page goes away", () => {
    const { unmount } = renderHook(() => useTabStatus({ analysing: false, answer: "O(n)" }));
    expect(document.title).toBe("O(n) · PolyO");
    unmount();
    expect(document.title).toBe(SITE_TITLE);
  });
});

describe("useFaviconSpinner", () => {
  let frame = 0;

  beforeEach(() => {
    vi.useFakeTimers();
    frame = 0;
    const ctx = new Proxy({}, { get: (_t, key) => (key === "createLinearGradient" ? () => ({ addColorStop() {} }) : () => {}), set: () => true });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockImplementation(() => `data:image/png;base64,FRAME${frame++}`);
    vi.stubGlobal("Path2D", class {});
    stubMotion(false);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    document.head.innerHTML = "";
  });

  it("leaves the icons alone while idle", () => {
    const links = addIconLinks();
    renderHook(() => useFaviconSpinner(false));
    vi.advanceTimersByTime(1000);
    expect(links[0].getAttribute("href")).toBe("/favicon.ico");
    expect(links[1].getAttribute("href")).toBe("/icon.svg?x");
  });

  it("paints a new frame into every tab icon while analysing, as png", () => {
    const links = addIconLinks();
    renderHook(() => useFaviconSpinner(true));
    const first = links[0].getAttribute("href");
    expect(first).toMatch(/^data:image\/png;base64,FRAME/);
    expect(links[1].getAttribute("href")).toBe(first); // the svg link too: no stale icon wins
    expect(links[1].getAttribute("type")).toBe("image/png");
    act(() => void vi.advanceTimersByTime(200));
    expect(links[0].getAttribute("href")).not.toBe(first);
  });

  it("does not touch the apple touch icon", () => {
    addIconLinks();
    renderHook(() => useFaviconSpinner(true));
    expect(document.querySelector('link[rel="apple-touch-icon"]')!.getAttribute("href")).toBe("/apple-icon.png");
  });

  it("restores every original href, type and size when the analysis ends", () => {
    const links = addIconLinks();
    const { rerender } = renderHook(({ on }) => useFaviconSpinner(on), { initialProps: { on: true } });
    rerender({ on: false });
    expect(links[0].getAttribute("href")).toBe("/favicon.ico");
    expect(links[0].getAttribute("sizes")).toBe("48x48");
    expect(links[0].hasAttribute("type")).toBe(false);
    expect(links[1].getAttribute("href")).toBe("/icon.svg?x");
    expect(links[1].getAttribute("type")).toBe("image/svg+xml");
    expect(links[1].getAttribute("sizes")).toBe("any");
    const after = links[0].getAttribute("href");
    vi.advanceTimersByTime(1000); // and it has really stopped
    expect(links[0].getAttribute("href")).toBe(after);
  });

  it("restores the icons if the page goes away mid-analysis", () => {
    const links = addIconLinks();
    const { unmount } = renderHook(() => useFaviconSpinner(true));
    unmount();
    expect(links[0].getAttribute("href")).toBe("/favicon.ico");
  });

  it("does not animate for a visitor who asked for reduced motion", () => {
    stubMotion(true);
    const links = addIconLinks();
    renderHook(() => useFaviconSpinner(true));
    vi.advanceTimersByTime(1000);
    expect(links[0].getAttribute("href")).toBe("/favicon.ico");
  });

  it("does nothing, and does not throw, with no icon links or no canvas support", () => {
    document.head.innerHTML = "";
    expect(() => renderHook(() => useFaviconSpinner(true))).not.toThrow();
    const links = addIconLinks();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    expect(() => renderHook(() => useFaviconSpinner(true))).not.toThrow();
    expect(links[0].getAttribute("href")).toBe("/favicon.ico");
  });
});
