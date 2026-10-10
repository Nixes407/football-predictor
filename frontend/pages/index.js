import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getApiBase } from '../lib/apiBase';
import Head from 'next/head';
import TodayTab from '../components/TodayTab';
import CustomTab from '../components/CustomTab';
import AboutTab from '../components/AboutTab';
import HistoryTab from '../components/HistoryTab';
import StatsTab from '../components/StatsTab';
import AccumulatorTab from '../components/AccumulatorTab';

const TABS = [
  { id: 0, label: '📅 Today' },
  { id: 1, label: '🔮 Custom' },
  { id: 2, label: '📋 History' },
  { id: 3, label: '📊 Stats' },
  { id: 4, label: '🎯 Acca' },
  { id: 5, label: 'ℹ️ About' },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState(0);

  // Warm up the backend as soon as the page loads (free-tier servers sleep).
  useEffect(() => {
    fetch(`${getApiBase()}/`).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen" style={{ background: 'var(--navy)' }}>
      <Head>
        <title>Football Predictor — ML Match Predictions</title>
        <meta
          name="description"
          content="Machine-learning football predictions for the top 5 European leagues"
        />
      </Head>

      <div className="max-w-6xl mx-auto px-4">
        {/* ── Header ───────────────────────────────────────────── */}
        <header className="pt-8 pb-4">
          <div className="flex items-end justify-between flex-wrap gap-2">
            <h1 className="text-3xl md:text-4xl font-bold flex items-center gap-3">
              <span className="float inline-block">⚽</span>
              <span className="gradient-text tracking-wide">
                FOOTBALL PREDICTOR
              </span>
            </h1>
            <span
              className="text-sm pb-1"
              style={{ color: 'var(--text-muted)' }}
            >
              Powered by ML
            </span>
          </div>
          <div
            className="mt-4 h-px w-full"
            style={{
              background:
                'linear-gradient(90deg, var(--blue), var(--green), transparent)',
            }}
          />
        </header>

        {/* ── Tab bar ──────────────────────────────────────────── */}
        <nav className="flex gap-1 mt-2 overflow-x-auto">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative px-5 py-3 text-sm md:text-base font-semibold rounded-t-lg transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? 'tab-active'
                  : 'hover:bg-white/5'
              }`}
              style={{
                color:
                  activeTab === tab.id ? 'var(--blue)' : 'var(--text-muted)',
              }}
            >
              {tab.label}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="tab-underline"
                  className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{
                    background:
                      'linear-gradient(90deg, var(--blue), var(--green))',
                  }}
                  transition={{ type: 'spring', stiffness: 400, damping: 35 }}
                />
              )}
            </button>
          ))}
        </nav>

        {/* ── Tab content ──────────────────────────────────────── */}
        <main className="py-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {activeTab === 0 && <TodayTab />}
              {activeTab === 1 && <CustomTab />}
              {activeTab === 2 && <HistoryTab />}
              {activeTab === 3 && <StatsTab />}
              {activeTab === 4 && <AccumulatorTab />}
              {activeTab === 5 && <AboutTab />}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
