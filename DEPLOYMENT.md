# Deployment Guide

**This guide has moved to [`docs/deploy.md`](docs/deploy.md).** That file covers
the current setup end to end: Vercel root directory, the frontend's
`NEXT_PUBLIC_API_URL`, the backend's CORS variables, the one-time data
pipeline, and a troubleshooting table.

Two things changed since this file was first written:

- **`vercel.json` no longer proxies `/api`.** Its `routes`/`builds` block was
  removed because it conflicted with Vercel's automatic Next.js detection. It
  now contains only `{ "version": 2 }`, and there is nothing in it to edit when
  the backend URL changes.
- **Backend URL lives in one place:** the `NEXT_PUBLIC_API_URL` environment
  variable on Vercel, normalised by `frontend/lib/apiBase.js`.

Because the browser now calls the backend directly instead of through a
same-origin proxy, the backend **must** allow the frontend origin through CORS
(`ALLOWED_ORIGINS` / `FRONTEND_URL`) — see [`docs/deploy.md`](docs/deploy.md).
