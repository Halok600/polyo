"use client";

// A thin hand-rolled CodeMirror 6 wrapper -- no @uiw/react-codemirror or
// basicSetup, both pull in more (search panels, fold gutters, a bundled
// theme) than this instrument-panel surface wants. Syntax highlighting is
// deliberately monochrome: globals.css's own rule is "hue carries meaning,
// nothing else gets a saturated color" (amber = model data, cyan = UI
// state), so tokens are told apart by weight/italic/opacity across the
// existing text-primary/secondary/muted scale, not a rainbow theme.

import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { bracketMatching, HighlightStyle, indentOnInput, StreamLanguage, syntaxHighlighting } from "@codemirror/language";
import { Compartment, EditorState, type Extension } from "@codemirror/state";
import { drawSelection, dropCursor, EditorView, type KeyBinding, keymap, lineNumbers } from "@codemirror/view";
import { tags } from "@lezer/highlight";
import { useEffect, useRef } from "react";

type CodeEditorProps = {
  value: string;
  onChange: (value: string) => void;
  language: string;
};

// Each served language's syntax package is only pulled in once someone
// actually selects it -- unlike the ~650KB CodeMirror core (see page.tsx's
// next/dynamic import of this component), these are small individually,
// but there's no reason to ship all five upfront when "auto" (no language
// package at all) is the default and most runs only ever use one language.
async function loadLanguageExtension(language: string): Promise<Extension | null> {
  switch (language) {
    case "python": {
      const { python } = await import("@codemirror/lang-python");
      return python();
    }
    case "javascript": {
      const { javascript } = await import("@codemirror/lang-javascript");
      return javascript();
    }
    case "java": {
      const { java } = await import("@codemirror/lang-java");
      return java();
    }
    case "cpp":
    case "c": {
      const { cpp } = await import("@codemirror/lang-cpp");
      return cpp();
    }
    case "go": {
      const { go } = await import("@codemirror/legacy-modes/mode/go");
      return StreamLanguage.define(go);
    }
    default:
      return null; // "auto" -- the language isn't known until a run completes.
  }
}

const benchHighlightStyle = HighlightStyle.define([
  { tag: [tags.comment, tags.lineComment, tags.blockComment], color: "var(--text-muted)", fontStyle: "italic" },
  { tag: [tags.keyword, tags.controlKeyword, tags.operatorKeyword, tags.definitionKeyword, tags.moduleKeyword], color: "var(--text-primary)", fontWeight: "700" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--text-secondary)" },
  { tag: tags.number, color: "var(--text-secondary)" },
  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: "var(--text-primary)" },
  { tag: [tags.typeName, tags.className], color: "var(--text-primary)", fontWeight: "600" },
  { tag: tags.propertyName, color: "var(--text-secondary)" },
  { tag: [tags.operator, tags.punctuation, tags.bracket], color: "var(--text-muted)" },
  { tag: tags.variableName, color: "var(--text-primary)" },
  { tag: tags.invalid, color: "var(--status-error)" },
]);

// @codemirror/commands' defaultKeymap binds Mod-Enter to insertBlankLine.
// page.tsx's own global Cmd/Ctrl+Enter (run analysis) also fires on this
// key -- CodeMirror's handler calls preventDefault() but not
// stopPropagation(), so without this the key would BOTH insert a blank
// line AND submit in the same keystroke. Listed ahead of defaultKeymap so
// it wins; returning true marks the key handled (blocking insertBlankLine)
// without doing anything else, leaving page.tsx's window listener as the
// sole owner of what Mod-Enter does.
const suppressModEnter: KeyBinding[] = [{ key: "Mod-Enter", run: () => true }];

const benchEditorTheme = EditorView.theme({
  "&": {
    backgroundColor: "var(--surface-2)",
    color: "var(--text-primary)",
    border: "1px solid var(--border)",
  },
  "&.cm-focused": {
    outline: "none",
  },
  ".cm-content": {
    fontFamily: "var(--font-mono-stack)",
    caretColor: "var(--signal)",
    padding: "12px 0",
  },
  ".cm-cursor, .cm-dropCursor": {
    borderLeftColor: "var(--signal)",
  },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection": {
    backgroundColor: "var(--signal-fill)",
  },
  ".cm-gutters": {
    backgroundColor: "var(--surface-2)",
    color: "var(--text-muted)",
    border: "none",
    borderRight: "1px solid var(--border)",
  },
  ".cm-activeLine, .cm-activeLineGutter": {
    backgroundColor: "transparent",
  },
  ".cm-scroller": {
    fontFamily: "var(--font-mono-stack)",
    lineHeight: "1.5",
  },
});

export function CodeEditor({ value, onChange, language }: CodeEditorProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const languageCompartment = useRef(new Compartment());
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });

  // Created once; the language and value effects below reconfigure/patch
  // the live view instead of tearing it down, so cursor position and undo
  // history survive a language switch.
  useEffect(() => {
    if (!containerRef.current) return;
    const view = new EditorView({
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(),
          history(),
          drawSelection(),
          dropCursor(),
          indentOnInput(),
          bracketMatching(),
          EditorView.lineWrapping,
          keymap.of([indentWithTab, ...suppressModEnter, ...defaultKeymap, ...historyKeymap]),
          // Starts empty -- the [language] effect below loads and applies
          // the initial language too, asynchronously, right after mount.
          languageCompartment.current.of([]),
          syntaxHighlighting(benchHighlightStyle),
          benchEditorTheme,
          EditorView.updateListener.of((update) => {
            if (update.docChanged) onChangeRef.current(update.state.doc.toString());
          }),
        ],
      }),
      parent: containerRef.current,
    });
    viewRef.current = view;
    return () => view.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deliberately mount-only; see comment above
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadLanguageExtension(language).then((extension) => {
      // Guards against a stale, out-of-order resolution -- rapidly
      // switching languages (or examples, which change language and code
      // together) must not let an earlier import "win" over the current
      // selection just because it happened to resolve later.
      if (cancelled) return;
      viewRef.current?.dispatch({ effects: languageCompartment.current.reconfigure(extension ?? []) });
    });
    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (current !== value) {
      view.dispatch({ changes: { from: 0, to: current.length, insert: value } });
    }
  }, [value]);

  // CodeMirror's own contentEditable already carries role="textbox" --
  // this wrapper is purely a layout/theming boundary, not a second control.
  return <div ref={containerRef} className="code-editor-frame" />;
}
