import { defineConfig } from 'vitest/config';
import { transformWithOxc } from 'vite';

export default defineConfig({
  // Next.js components use JSX in .js files rather than a .jsx extension.
  plugins: [{
    name: 'test-jsx-in-js',
    enforce: 'pre',
    async transform(code, id) {
      if (/\/components\/.*\.js$/.test(id)) {
        return transformWithOxc(code, id, { lang: 'jsx', jsx: { runtime: 'automatic' } });
      }
    },
  }],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.js'],
  },
});
