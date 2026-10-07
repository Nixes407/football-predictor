import '../styles/globals.css';

/**
 * Custom App component.
 *
 * Next.js only allows global stylesheets to be imported here, so this file is
 * what loads Tailwind for every page.
 */
export default function App({ Component, pageProps }) {
  return <Component {...pageProps} />;
}
