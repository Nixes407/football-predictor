import Head from 'next/head';

/**
 * Landing page placeholder.
 *
 * Phase 1 scaffold only — no prediction logic, no API calls, no forms yet.
 */
export default function Home() {
  return (
    <>
      <Head>
        <title>Football Match Predictor</title>
        <meta
          name="description"
          content="Predict the outcome of football matches."
        />
      </Head>

      <main className="flex min-h-screen flex-col items-center justify-center bg-white px-6 text-center">
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl">
          Football Match Predictor
        </h1>
        <p className="mt-4 text-lg text-slate-500">Loading soon...</p>
      </main>
    </>
  );
}
