import type { LanguageOption } from "@/lib/types";

type LanguageSelectorProps = {
  languages: LanguageOption[];
  value: string;
  onChange: (value: string) => void;
};

export function LanguageSelector({ languages, value, onChange }: LanguageSelectorProps) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 13 }}>
      <span style={{ color: "var(--text-muted)" }}>Language</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        style={{
          padding: "8px 10px",
          borderRadius: 8,
          border: "1px solid var(--border)",
          background: "var(--surface-1)",
          color: "var(--text-primary)",
          fontSize: 14,
        }}
      >
        <option value="auto">Auto-detect</option>
        {languages.map((lang) => (
          <option key={lang.id} value={lang.id}>
            {lang.id}
            {lang.tier === 2 ? " (partial support)" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}
