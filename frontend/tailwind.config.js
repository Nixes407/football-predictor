/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: '#0a0f1e',
          light: '#111827',
          card: '#0d1526',
        },
        electric: {
          blue: '#00d4ff',
          green: '#00ff88',
        },
      },
      fontFamily: {
        sans: ['Space Grotesk', 'sans-serif'],
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
