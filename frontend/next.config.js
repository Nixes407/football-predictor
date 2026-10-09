/**
 * Next.js configuration.
 *
 * No `/api/*` rewrite is defined here on purpose: the browser calls the
 * FastAPI backend directly, using `NEXT_PUBLIC_API_URL` resolved in
 * `lib/apiBase.js`. That keeps a single request path with no proxy layer —
 * neither `vercel.json` routes nor a Next.js rewrite is involved.
 *
 * Adding a rewrite back would mean the request only works on the Node server
 * (`next start` / Vercel's Node runtime) and not from a static export, and it
 * would reintroduce the same class of proxy conflict that `vercel.json` used
 * to cause. If you do need one, point it at the same env var, e.g.
 *   destination: `${process.env.NEXT_PUBLIC_API_URL}/:path*`
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
