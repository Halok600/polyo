import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Vitest runs without globals, so Testing Library cannot register its own auto-cleanup: without
// this the DOM of one test is still there in the next and queries find several matches.
afterEach(() => {
  cleanup();
});
