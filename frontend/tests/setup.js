import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// Framer Motion restores scroll position after measuring animated elements.
window.scrollTo = () => {};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();
});
