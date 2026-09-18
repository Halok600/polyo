"use client";

// Custom role="listbox" combobox -- a native <select>'s open/close can't be
// animated across browsers, so this hand-rolls the interaction (button +
// floating panel) to get the scale+fade open animation while keeping
// keyboard support (arrows/enter/escape) and listbox semantics.

import { useEffect, useId, useRef, useState } from "react";

import type { LanguageOption } from "@/lib/types";

type LanguageSelectorProps = {
  languages: LanguageOption[];
  value: string;
  onChange: (value: string) => void;
};

type Option = { id: string; label: string; tier?: 1 | 2 };

export function LanguageSelector({ languages, value, onChange }: LanguageSelectorProps) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);
  const labelId = useId();

  const options: Option[] = [{ id: "auto", label: "auto-detect" }, ...languages.map((lang) => ({ id: lang.id, label: lang.id, tier: lang.tier }))];
  const selectedIndex = Math.max(0, options.findIndex((opt) => opt.id === value));
  const selected = options[selectedIndex] ?? options[0];

  useEffect(() => {
    if (!open) return;
    listRef.current?.focus();

    function handlePointerDown(event: PointerEvent) {
      if (!listRef.current?.contains(event.target as Node) && !buttonRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [open]);

  function commit(index: number) {
    onChange(options[index].id);
    setOpen(false);
    buttonRef.current?.focus();
  }

  function openPanel() {
    setActiveIndex(selectedIndex);
    setOpen(true);
  }

  function handleButtonKeyDown(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPanel();
    }
  }

  function handleListKeyDown(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((i) => Math.min(options.length - 1, i + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      commit(activeIndex);
    } else if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      buttonRef.current?.focus();
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 13, position: "relative" }}>
      <span id={labelId} className="mono-nums" style={{ color: "var(--text-muted)", fontSize: 11, letterSpacing: "0.08em" }}>
        LANG
      </span>
      <button
        type="button"
        ref={buttonRef}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby={labelId}
        onClick={() => (open ? setOpen(false) : openPanel())}
        onKeyDown={handleButtonKeyDown}
        className="mono-nums lang-select-trigger"
      >
        <span>
          {selected.label}
          {selected.tier === 2 ? " (partial)" : ""}
        </span>
        <span aria-hidden className={`lang-select-caret${open ? " is-open" : ""}`}>
          ▾
        </span>
      </button>
      {open ? (
        <ul ref={listRef} role="listbox" aria-labelledby={labelId} tabIndex={-1} onKeyDown={handleListKeyDown} className="mono-nums lang-select-panel">
          {options.map((opt, i) => (
            <li
              key={opt.id}
              role="option"
              aria-selected={opt.id === value}
              onMouseEnter={() => setActiveIndex(i)}
              onClick={() => commit(i)}
              className={`lang-select-option${i === activeIndex ? " is-active" : ""}`}
            >
              <span aria-hidden className="lang-select-marker">
                {opt.id === value ? "▸" : ""}
              </span>
              {opt.label}
              {opt.tier === 2 ? <span className="lang-select-badge">partial</span> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
