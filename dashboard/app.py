"""
dashboard/app.py
----------------
FastAPI backend for the RiskAwareRL Portfolio Dashboard.
Pre-runs all three agents on demo_dataset.csv, then streams
per-step results via SSE so the frontend animates in real-time.

Run:
    uvicorn dashboard.app:app --port 8000
"""

import asyncio, json, os, sys, warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.trading_environment import PortfolioTradingEnv, ASSETS
from agents.train_dqn import DiscretePortfolioWrapper
from data_pipeline.regime_detection import detect_market_regime

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_dataset.csv")

AGENTS_CFG = {
    "DQN": {
        "algo": "DQN", "model": "dqn_portfolio", "vecnorm": None,
        "safe": False, "color": "#f59e0b", "rgb": "245,158,11", "tag": "Discrete",
    },
    "PPO": {
        "algo": "PPO", "model": "ppo_portfolio",
        "vecnorm": "ppo_portfolio_vecnorm.pkl",
        "safe": False, "color": "#60a5fa", "rgb": "96,165,250", "tag": "Continuous",
    },
    "Safe PPO": {
        "algo": "PPO", "model": "safe_ppo_portfolio",
        "vecnorm": "safe_ppo_portfolio_vecnorm.pkl",
        "safe": True, "color": "#34d399", "rgb": "52,211,153", "tag": "Risk-Aware",
    },
}

app = FastAPI(title="RiskAwareRL Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

SIM: dict = {"steps": [], "metrics": {}, "ready": False}


# ── Agent runner ──────────────────────────────────────────────────────────────

def _run_agent(name: str, cfg: dict, df: pd.DataFrame) -> list:
    AlgoCls = DQN if cfg["algo"] == "DQN" else PPO
    model = AlgoCls.load(os.path.join(MODEL_DIR, cfg["model"]))

    vecnorm = None
    if cfg.get("vecnorm"):
        vp = os.path.join(MODEL_DIR, cfg["vecnorm"])
        if os.path.isfile(vp):
            dummy = DummyVecEnv([lambda: Monitor(
                PortfolioTradingEnv(df, safe_reward=cfg["safe"], random_start=False)
            )])
            vecnorm = VecNormalize.load(vp, dummy)
            vecnorm.training = False
            vecnorm.norm_reward = False

    base = PortfolioTradingEnv(df, safe_reward=cfg["safe"], random_start=False)
    env  = (DiscretePortfolioWrapper(base, np.load(os.path.join(MODEL_DIR, "dqn_action_table.npy")))
            if cfg["algo"] == "DQN" else base)

    obs, _ = env.reset()
    done   = False
    iv     = base.initial_value
    peak   = iv
    steps  = [{"v": iv, "ret": 0.0, "dd": 0.0,
                "w": [round(float(x), 3) for x in base.weights],
                "r": 0, "dr": 0.0}]

    while not done:
        oi = vecnorm.normalize_obs(obs.reshape(1, -1))[0] if vecnorm else obs
        a, _ = model.predict(oi, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        done = done or trunc
        pv   = float(info["portfolio_value"])
        peak = max(peak, pv)
        dd   = (peak - pv) / peak if peak > 0 else 0.0
        steps.append({
            "v":   round(pv, 2),
            "ret": round((pv - iv) / iv * 100, 3),
            "dd":  round(dd * 100, 3),
            "w":   [round(float(x), 3) for x in info["weights"]],
            "r":   int(info.get("regime", 0)),
            "dr":  round(float(info.get("net_return", 0)) * 100, 4),
        })
    return steps


def _metrics(steps: list) -> dict:
    vals = [s["v"]  for s in steps]
    dr   = np.array([s["dr"] for s in steps[1:]])
    tr   = (vals[-1] - vals[0]) / vals[0] * 100
    std  = dr.std()
    sh   = float(dr.mean() / std  * np.sqrt(252)) if std > 1e-10  else 0.0
    neg  = dr[dr < 0]
    ds   = neg.std() if len(neg) > 0 else 1e-10
    so   = float(dr.mean() / ds   * np.sqrt(252)) if ds  > 1e-10  else 0.0
    mdd  = max(s["dd"] for s in steps)
    return {"tr": round(tr,2), "fv": round(vals[-1],2),
            "sh": round(sh,3), "so": round(so,3), "mdd": round(mdd,2)}


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    print("Loading test dataset & running agents…")
    df    = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df    = detect_market_regime(df)
    dates = [str(d.date()) for d in df.index]

    # Override the environment's episode length so the full dataset is
    # simulated (default MIN_EPISODE_STEPS=252 caps at ~1 year).
    import env.trading_environment as _te
    _te.MIN_EPISODE_STEPS = len(df) - 1

    all_steps: dict = {}
    for name, cfg in AGENTS_CFG.items():
        mp = os.path.join(MODEL_DIR, cfg["model"] + ".zip")
        if not os.path.isfile(mp):
            print(f"  Skipping {name}: {mp} not found")
            continue
        print(f"  Running {name}…")
        all_steps[name] = _run_agent(name, cfg, df)
        print(f"  {name}: {len(all_steps[name])} steps")

    if not all_steps:
        return

    n = min(len(v) for v in all_steps.values())
    SIM["steps"] = [
        {"i": i, "d": dates[min(i, len(dates) - 1)],
         "a": {nm: data[i] for nm, data in all_steps.items() if i < len(data)}}
        for i in range(n)
    ]
    SIM["metrics"] = {nm: _metrics(steps) for nm, steps in all_steps.items()}
    SIM["ready"]   = True
    print(f"Dashboard ready — {n} steps, agents: {list(all_steps.keys())}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_html())


@app.get("/api/run")
async def run_sim(delay: float = 0.10):
    if not SIM["ready"]:
        async def err():
            yield 'data: {"error":"Not ready"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    async def gen():
        yield "data: " + json.dumps({
            "init": True, "meta": SIM["metrics"], "n": len(SIM["steps"])
        }) + "\n\n"
        await asyncio.sleep(0.05)
        for step in SIM["steps"]:
            yield "data: " + json.dumps(step) + "\n\n"
            await asyncio.sleep(max(0.003, delay))
        yield 'data: {"done":true}\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# ── HTML builder ──────────────────────────────────────────────────────────────

def _html() -> str:
    agents_js = json.dumps({
        k: {"color": v["color"], "rgb": v["rgb"], "tag": v["tag"]}
        for k, v in AGENTS_CFG.items()
    })
    assets_js = json.dumps(ASSETS)
    return (_HTML_TEMPLATE
            .replace("__AGENTS_CFG__", agents_js)
            .replace("__ASSET_NAMES__", assets_js))


# ─────────────────────────────────────────────────────────────────────────────
#  HTML / CSS / JS  (plain string — no f-string escaping needed)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>RiskAwareRL — Portfolio Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #04080f;
  --bg2:     #070e1a;
  --surface: #0b1425;
  --surf2:   #0f1c33;
  --border:  #162845;
  --bord2:   #1e3355;
  --text:    #e2e8f0;
  --muted:   #546a8a;
  --dim:     #2a4060;
  --dqn:     #f59e0b;
  --ppo:     #60a5fa;
  --safe:    #34d399;
  --bull:    #10b981;
  --neut:    #64748b;
  --bear:    #ef4444;
  --green:   #4ade80;
  --red:     #f87171;
}

html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: 'Inter', sans-serif; font-size: 14px; overflow-x: hidden; }

::-webkit-scrollbar { width: 6px; background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--bord2); border-radius: 3px; }

.app { max-width: 1600px; margin: 0 auto; padding: 16px;
       display: flex; flex-direction: column; gap: 14px; }

/* ── Header ── */
.header {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 24px; gap: 16px;
  box-shadow: 0 0 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-icon {
  width: 42px; height: 42px; border-radius: 10px;
  background: linear-gradient(135deg, #1d4ed8, #7c3aed);
  display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.brand-name { font-size: 18px; font-weight: 800; letter-spacing: -0.5px; }
.brand-name span { color: #60a5fa; }
.brand-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

.hdr-center { display: flex; align-items: center; gap: 24px; }
.live-wrap { display: flex; align-items: center; gap: 8px; }
.live-dot {
  width: 9px; height: 9px; border-radius: 50%; background: #10b981;
  box-shadow: 0 0 10px #10b981; animation: pulse 1.4s ease-in-out infinite;
}
.live-dot.idle { background: var(--muted); box-shadow: none; animation: none; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.8); } }
.live-label { font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #10b981; }
.live-label.idle { color: var(--muted); }

.sim-date-wrap { text-align: center; }
.sim-date-lbl { font-size: 10px; color: var(--muted); letter-spacing: 0.8px; text-transform: uppercase; }
.sim-date-val { font-family: 'JetBrains Mono', monospace; font-size: 16px;
                font-weight: 600; letter-spacing: 0.5px; margin-top: 2px; }

.regime-badge { padding: 4px 12px; border-radius: 20px; font-size: 11px;
                font-weight: 700; letter-spacing: 1px; transition: all 0.4s; }
.regime-badge.bull    { background: rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3); }
.regime-badge.neutral { background: rgba(100,116,139,0.15);color:#94a3b8; border:1px solid rgba(100,116,139,0.3); }
.regime-badge.bear    { background: rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.3); }

.hdr-right { text-align: right; }
.step-lbl { font-size: 10px; color: var(--muted); letter-spacing: 0.8px; text-transform: uppercase; }
.step-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; margin-top: 2px; }

/* ── Controls bar ── */
.controls-bar {
  display: flex; align-items: center; gap: 16px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 20px;
  box-shadow: 0 0 40px rgba(0,0,0,0.4);
}
.play-btn {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 22px; border-radius: 8px; border: none; cursor: pointer;
  font-size: 13px; font-weight: 700; letter-spacing: 0.4px; white-space: nowrap;
  background: linear-gradient(135deg, #1d4ed8, #7c3aed); color: #fff;
  transition: all 0.2s; box-shadow: 0 0 20px rgba(96,165,250,0.2);
}
.play-btn:hover { transform: translateY(-1px); box-shadow: 0 0 30px rgba(96,165,250,0.35); }
.play-btn.stop   { background: linear-gradient(135deg, #7f1d1d, #ef4444); box-shadow: 0 0 20px rgba(239,68,68,0.2); }
.play-btn.replay { background: linear-gradient(135deg, #14532d, #22c55e); box-shadow: 0 0 20px rgba(34,197,94,0.2); }

.speed-group { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.speed-lbl { font-size: 11px; color: var(--muted); font-weight: 500;
             text-transform: uppercase; letter-spacing: 0.8px; }
.speed-pills { display: flex; gap: 4px; }
.pill { padding: 5px 12px; border-radius: 6px; border: 1px solid var(--bord2);
        background: transparent; color: var(--muted); font-size: 11px;
        font-weight: 600; cursor: pointer; transition: all 0.15s; letter-spacing: 0.3px; }
.pill:hover { border-color: var(--ppo); color: var(--text); }
.pill.active { background: rgba(96,165,250,0.15); border-color: var(--ppo); color: var(--ppo); }

.progress-outer { flex: 1; height: 6px; background: var(--surf2);
                  border-radius: 3px; overflow: hidden; border: 1px solid var(--border); }
.progress-fill { height: 100%; width: 0%; border-radius: 3px;
                 background: linear-gradient(90deg, #1d4ed8, #7c3aed, #60a5fa);
                 transition: width 0.3s ease; }
.progress-pct { font-family: 'JetBrains Mono', monospace; font-size: 12px;
                font-weight: 600; color: var(--muted); min-width: 36px; text-align: right; }

/* ── Agent Cards ── */
.cards-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }
.agent-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; position: relative; overflow: hidden;
  box-shadow: 0 0 40px rgba(0,0,0,0.4); transition: border-color 0.3s;
}
.agent-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px; border-radius: 12px 12px 0 0;
}
.card-dqn::before  { background: var(--dqn); }
.card-ppo::before  { background: var(--ppo); }
.card-safe::before { background: var(--safe); }

.card-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.agent-label { display: flex; align-items: center; gap: 8px; }
.adot { width: 10px; height: 10px; border-radius: 50%; }
.adot-dqn  { background: var(--dqn); box-shadow: 0 0 8px var(--dqn); }
.adot-ppo  { background: var(--ppo); box-shadow: 0 0 8px var(--ppo); }
.adot-safe { background: var(--safe);box-shadow: 0 0 8px var(--safe);}
.agent-name-txt { font-size: 14px; font-weight: 700; }
.agent-tag { padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; letter-spacing: 0.4px; }
.card-dqn  .agent-tag { background: rgba(245,158,11,0.15); color: var(--dqn); }
.card-ppo  .agent-tag { background: rgba(96,165,250,0.15); color: var(--ppo); }
.card-safe .agent-tag { background: rgba(52,211,153,0.15); color: var(--safe); }

.card-value { font-family: 'JetBrains Mono', monospace; font-size: 26px;
              font-weight: 600; letter-spacing: -0.5px; margin-bottom: 4px; }
.card-dqn  .card-value { color: var(--dqn); }
.card-ppo  .card-value { color: var(--ppo); }
.card-safe .card-value { color: var(--safe); }

.card-return { font-size: 15px; font-weight: 700; margin-bottom: 16px; }
.card-return.pos  { color: var(--green); }
.card-return.neg  { color: var(--red); }
.card-return.zero { color: var(--muted); }

.card-stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-bottom: 14px; }
.stat-box { background: var(--surf2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.stat-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase;
            letter-spacing: 0.6px; margin-bottom: 3px; }
.stat-num { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; }
.sparkline-wrap { height: 52px; }

/* ── Chart cards ── */
.charts-main      { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; }
.charts-secondary { display: grid; grid-template-columns: 1fr 1fr;  gap: 14px; }
.chart-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px;
  box-shadow: 0 0 40px rgba(0,0,0,0.4);
}
.chart-hdr { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.chart-title { font-size: 13px; font-weight: 700; }
.chart-legend { display: flex; align-items: center; gap: 16px; }
.leg-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }
.leg-line { width: 22px; height: 2.5px; border-radius: 2px; }
.chart-body { position: relative; height: 255px; }
.alloc-date { font-size: 11px; color: var(--muted); }

/* ── Metrics table ── */
.metrics-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
  box-shadow: 0 0 40px rgba(0,0,0,0.4);
}
.complete-badge { padding: 4px 12px; border-radius: 20px; font-size: 11px;
  font-weight: 700; letter-spacing: 1px;
  background: rgba(52,211,153,0.15); color: var(--safe); border: 1px solid rgba(52,211,153,0.3); }
.mtable { width: 100%; border-collapse: collapse; margin-top: 4px; }
.mtable th { text-align: left; font-size: 10px; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted);
             padding: 10px 14px; border-bottom: 1px solid var(--border); }
.mtable td { padding: 12px 14px; border-bottom: 1px solid rgba(22,40,69,0.5);
             font-size: 13px; font-family: 'JetBrains Mono', monospace; }
.mtable tr:last-child td { border-bottom: none; }
.mtable tr:hover td { background: rgba(15,28,51,0.8); }
.agent-cell { display: flex; align-items: center; gap: 9px; }
.aswatch { width: 10px; height: 10px; border-radius: 50%; }
.acell-name { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13px; }
.best  { color: var(--green) !important; font-weight: 700; }
.worst { color: var(--red); }

.hidden { display: none !important; }

@media (max-width: 1100px) {
  .charts-main { grid-template-columns: 1fr; }
  .charts-secondary { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .cards-row { grid-template-columns: 1fr; }
  .hdr-center { display: none; }
}
</style>
</head>
<body>
<div class="app">

<!-- HEADER -->
<header class="header">
  <div class="brand">
    <div class="brand-icon">📈</div>
    <div>
      <div class="brand-name">RiskAware<span>RL</span></div>
      <div class="brand-sub">Safe Reinforcement Learning &nbsp;·&nbsp; Portfolio Management</div>
    </div>
  </div>
  <div class="hdr-center">
    <div class="live-wrap">
      <div id="liveDot" class="live-dot idle"></div>
      <span id="liveLabel" class="live-label idle">READY</span>
    </div>
    <div class="sim-date-wrap">
      <div class="sim-date-lbl">Simulated Date</div>
      <div id="currentDate" class="sim-date-val">──────────</div>
    </div>
    <div id="regimeBadge" class="regime-badge neutral">● NEUTRAL</div>
  </div>
  <div class="hdr-right">
    <div class="step-lbl">Progress</div>
    <div id="stepDisplay" class="step-val">0 / 0</div>
  </div>
</header>

<!-- CONTROLS -->
<div class="controls-bar">
  <button id="playBtn" class="play-btn" onclick="handlePlay()">
    <span id="playIcon">▶</span>
    <span id="playLabel">START SIMULATION</span>
  </button>
  <div class="speed-group">
    <span class="speed-lbl">Speed</span>
    <div class="speed-pills">
      <button class="pill" data-delay="0.30" onclick="setSpeed(this)">Slow</button>
      <button class="pill active" data-delay="0.10" onclick="setSpeed(this)">Normal</button>
      <button class="pill" data-delay="0.025" onclick="setSpeed(this)">Fast</button>
      <button class="pill" data-delay="0.004" onclick="setSpeed(this)">Turbo</button>
    </div>
  </div>
  <div class="progress-outer">
    <div id="progressFill" class="progress-fill"></div>
  </div>
  <span id="progressPct" class="progress-pct">0%</span>
</div>

<!-- AGENT CARDS -->
<div class="cards-row" id="cardsRow"></div>

<!-- MAIN CHARTS -->
<div class="charts-main">
  <div class="chart-card">
    <div class="chart-hdr">
      <span class="chart-title">Portfolio Value</span>
      <div class="chart-legend" id="mainLegend"></div>
    </div>
    <div class="chart-body"><canvas id="portfolioChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-hdr">
      <span class="chart-title">Drawdown (%)</span>
      <div class="chart-legend" id="ddLegend"></div>
    </div>
    <div class="chart-body"><canvas id="drawdownChart"></canvas></div>
  </div>
</div>

<!-- SECONDARY CHARTS -->
<div class="charts-secondary">
  <div class="chart-card">
    <div class="chart-hdr">
      <span class="chart-title">Cumulative Return (%)</span>
      <div class="chart-legend" id="retLegend"></div>
    </div>
    <div class="chart-body"><canvas id="returnChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-hdr">
      <span class="chart-title">Current Portfolio Allocation</span>
      <span id="allocDate" class="alloc-date"></span>
    </div>
    <div class="chart-body"><canvas id="allocChart"></canvas></div>
  </div>
</div>

<!-- METRICS TABLE -->
<div id="metricsSection" class="metrics-card hidden">
  <div class="chart-hdr">
    <span class="chart-title">Final Performance Comparison</span>
    <span class="complete-badge">✓ SIMULATION COMPLETE</span>
  </div>
  <table class="mtable" id="metricsTable"></table>
</div>

</div><!-- /app -->

<script>
// ── Config (injected by server) ───────────────────────────────────────────
const AGENTS_CFG  = __AGENTS_CFG__;
const ASSET_NAMES = __ASSET_NAMES__;
const AGENT_KEYS  = Object.keys(AGENTS_CFG);

// ── State ─────────────────────────────────────────────────────────────────
let es = null, isRunning = false, isDone = false;
let totalSteps = 0, currentStep = 0, currentDelay = 0.10;
let metaMetrics = {};
let portfolioChart, drawdownChart, returnChart, allocChart;
const sparkCharts = {};

// ── Helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt    = v => '$' + Number(v).toLocaleString('en-US',
                   { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = v => (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';

// ── Build cards ───────────────────────────────────────────────────────────
function buildCards() {
  const row = $('cardsRow');
  AGENT_KEYS.forEach(name => {
    const cfg  = AGENTS_CFG[name];
    const slug = name.replace(/\s+/g, '_');
    const cardCls = name === 'DQN' ? 'card-dqn' : name === 'PPO' ? 'card-ppo' : 'card-safe';
    const dotCls  = name === 'DQN' ? 'adot-dqn' : name === 'PPO' ? 'adot-ppo' : 'adot-safe';
    const div = document.createElement('div');
    div.className = 'agent-card ' + cardCls;
    div.id = 'card-' + slug;
    div.innerHTML =
      '<div class="card-top">' +
        '<div class="agent-label">' +
          '<div class="adot ' + dotCls + '"></div>' +
          '<span class="agent-name-txt">' + name + '</span>' +
        '</div>' +
        '<span class="agent-tag">' + cfg.tag + '</span>' +
      '</div>' +
      '<div id="val-' + slug + '" class="card-value">$100,000.00</div>' +
      '<div id="ret-' + slug + '" class="card-return zero">+0.00%</div>' +
      '<div class="card-stats">' +
        '<div class="stat-box"><div class="stat-lbl">Sharpe</div>' +
          '<div id="sh-' + slug + '" class="stat-num">—</div></div>' +
        '<div class="stat-box"><div class="stat-lbl">Max DD</div>' +
          '<div id="dd-' + slug + '" class="stat-num">—</div></div>' +
        '<div class="stat-box"><div class="stat-lbl">Sortino</div>' +
          '<div id="so-' + slug + '" class="stat-num">—</div></div>' +
      '</div>' +
      '<div class="sparkline-wrap"><canvas id="spark-' + slug + '" height="52"></canvas></div>';
    row.appendChild(div);
  });
  requestAnimationFrame(initSparklines);
}

function buildLegends() {
  ['mainLegend','ddLegend','retLegend'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.innerHTML = AGENT_KEYS.map(n =>
      '<div class="leg-item">' +
        '<div class="leg-line" style="background:' + AGENTS_CFG[n].color + '"></div>' +
        n +
      '</div>'
    ).join('');
  });
}

// ── Chart init ────────────────────────────────────────────────────────────
const BASE_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: {
    legend: { display: false },
    tooltip: {
      mode: 'index', intersect: false,
      backgroundColor: 'rgba(7,14,26,0.95)',
      borderColor: 'rgba(26,40,69,0.9)', borderWidth: 1, padding: 10,
      titleColor: '#94a3b8', bodyColor: '#e2e8f0',
    }
  },
  elements: { point: { radius: 0, hoverRadius: 5 } },
  scales: {
    x: { grid: { color: 'rgba(22,40,69,0.45)' },
         ticks: { color: '#546a8a', maxTicksLimit: 8, font: { size: 11 } } },
    y: { grid: { color: 'rgba(22,40,69,0.45)' },
         ticks: { color: '#546a8a', font: { size: 11 } } }
  }
};

function mkDataset(name, extra) {
  const cfg = AGENTS_CFG[name];
  return Object.assign({
    label: name, data: [],
    borderColor: cfg.color,
    backgroundColor: 'rgba(' + cfg.rgb + ',0.06)',
    borderWidth: 2.5, tension: 0.3, fill: false,
    pointRadius: 0, pointHoverRadius: 5
  }, extra || {});
}

function initCharts() {
  // Portfolio Value
  portfolioChart = new Chart($('portfolioChart'), {
    type: 'line',
    data: { labels: [], datasets: AGENT_KEYS.map(n => mkDataset(n)) },
    options: Object.assign({}, BASE_OPTS, {
      plugins: Object.assign({}, BASE_OPTS.plugins, {
        tooltip: Object.assign({}, BASE_OPTS.plugins.tooltip, {
          callbacks: { label: c => ' ' + c.dataset.label + ': ' + fmt(c.raw) }
        })
      }),
      scales: Object.assign({}, BASE_OPTS.scales, {
        y: Object.assign({}, BASE_OPTS.scales.y, {
          ticks: Object.assign({}, BASE_OPTS.scales.y.ticks, {
            callback: v => '$' + Number(v / 1000).toFixed(0) + 'k'
          })
        })
      })
    })
  });

  // Drawdown
  drawdownChart = new Chart($('drawdownChart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: AGENT_KEYS.map(n => mkDataset(n, {
        fill: true,
        backgroundColor: 'rgba(' + AGENTS_CFG[n].rgb + ',0.12)'
      }))
    },
    options: Object.assign({}, BASE_OPTS, {
      plugins: Object.assign({}, BASE_OPTS.plugins, {
        tooltip: Object.assign({}, BASE_OPTS.plugins.tooltip, {
          callbacks: { label: c => ' ' + c.dataset.label + ': -' + Math.abs(c.raw).toFixed(2) + '%' }
        })
      }),
      scales: Object.assign({}, BASE_OPTS.scales, {
        y: Object.assign({}, BASE_OPTS.scales.y, {
          max: 0,
          ticks: Object.assign({}, BASE_OPTS.scales.y.ticks, {
            callback: v => v.toFixed(1) + '%'
          })
        })
      })
    })
  });

  // Cumulative Return
  returnChart = new Chart($('returnChart'), {
    type: 'line',
    data: { labels: [], datasets: AGENT_KEYS.map(n => mkDataset(n)) },
    options: Object.assign({}, BASE_OPTS, {
      plugins: Object.assign({}, BASE_OPTS.plugins, {
        tooltip: Object.assign({}, BASE_OPTS.plugins.tooltip, {
          callbacks: { label: c => ' ' + c.dataset.label + ': ' + fmtPct(c.raw) }
        })
      }),
      scales: Object.assign({}, BASE_OPTS.scales, {
        y: Object.assign({}, BASE_OPTS.scales.y, {
          ticks: Object.assign({}, BASE_OPTS.scales.y.ticks, {
            callback: v => fmtPct(v)
          })
        })
      })
    })
  });

  // Allocation bar
  allocChart = new Chart($('allocChart'), {
    type: 'bar',
    data: {
      labels: ASSET_NAMES,
      datasets: AGENT_KEYS.map(n => ({
        label: n, data: new Array(ASSET_NAMES.length).fill(1 / ASSET_NAMES.length),
        backgroundColor: 'rgba(' + AGENTS_CFG[n].rgb + ',0.72)',
        borderColor: AGENTS_CFG[n].color, borderWidth: 1, borderRadius: 3
      }))
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#546a8a', font: { size: 11 }, boxWidth: 12, boxHeight: 12 }
        },
        tooltip: {
          backgroundColor: 'rgba(7,14,26,0.95)', borderColor: 'rgba(26,40,69,0.9)', borderWidth: 1,
          callbacks: { label: c => ' ' + c.dataset.label + ': ' + (c.raw * 100).toFixed(1) + '%' }
        }
      },
      scales: {
        x: {
          stacked: false, grid: { color: 'rgba(22,40,69,0.45)' }, max: 0.55,
          ticks: { color: '#546a8a', font: { size: 11 }, callback: v => (v * 100).toFixed(0) + '%' }
        },
        y: { stacked: false, grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 11 } } }
      }
    }
  });
}

function initSparklines() {
  AGENT_KEYS.forEach(name => {
    const slug = name.replace(/\s+/g, '_');
    const canvas = $('spark-' + slug);
    if (!canvas) return;
    sparkCharts[name] = new Chart(canvas, {
      type: 'line',
      data: {
        labels: [''],
        datasets: [{
          data: [100000],
          borderColor: AGENTS_CFG[name].color, borderWidth: 1.5,
          tension: 0.3, fill: true,
          backgroundColor: 'rgba(' + AGENTS_CFG[name].rgb + ',0.12)',
          pointRadius: 0
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });
  });
}

// ── Step update ───────────────────────────────────────────────────────────
function applyStep(step) {
  currentStep = step.i;
  const pct = totalSteps > 1 ? (currentStep / (totalSteps - 1) * 100) : 0;

  $('currentDate').textContent   = step.d;
  $('stepDisplay').textContent   = currentStep + ' / ' + totalSteps;
  $('progressFill').style.width  = pct + '%';
  $('progressPct').textContent   = pct.toFixed(0) + '%';

  // Regime from first agent
  const firstKey = AGENT_KEYS.find(k => step.a && step.a[k]);
  if (firstKey) {
    const r  = step.a[firstKey].r;
    const rb = $('regimeBadge');
    if (r === 1)  { rb.className = 'regime-badge bull';    rb.textContent = '● BULL'; }
    else if (r === -1) { rb.className = 'regime-badge bear'; rb.textContent = '● BEAR'; }
    else          { rb.className = 'regime-badge neutral'; rb.textContent = '● NEUTRAL'; }
  }

  // Push to main charts
  portfolioChart.data.labels.push(step.d);
  drawdownChart.data.labels.push(step.d);
  returnChart.data.labels.push(step.d);

  AGENT_KEYS.forEach((name, idx) => {
    const d = step.a && step.a[name];
    if (!d) return;

    portfolioChart.data.datasets[idx].data.push(d.v);
    drawdownChart.data.datasets[idx].data.push(-d.dd);
    returnChart.data.datasets[idx].data.push(d.ret);

    // Sparkline
    const sp = sparkCharts[name];
    if (sp) {
      sp.data.labels.push(step.d);
      sp.data.datasets[0].data.push(d.v);
      sp.update('none');
    }

    // Allocation
    if (allocChart.data.datasets[idx]) {
      allocChart.data.datasets[idx].data = d.w;
    }

    // Card
    const slug = name.replace(/\s+/g, '_');
    $('val-' + slug).textContent = fmt(d.v);
    const retEl = $('ret-' + slug);
    retEl.textContent = fmtPct(d.ret);
    retEl.className   = 'card-return ' + (d.ret > 0.01 ? 'pos' : d.ret < -0.01 ? 'neg' : 'zero');
  });

  portfolioChart.update('none');
  drawdownChart.update('none');
  returnChart.update('none');
  allocChart.data.labels = ASSET_NAMES;
  allocChart.update();
  $('allocDate').textContent = step.d;
}

// ── Metrics table ─────────────────────────────────────────────────────────
function showMetrics() {
  if (!Object.keys(metaMetrics).length) return;
  const section = $('metricsSection');
  section.classList.remove('hidden');

  AGENT_KEYS.forEach(name => {
    const m = metaMetrics[name];
    if (!m) return;
    const slug = name.replace(/\s+/g, '_');
    $('sh-' + slug).textContent = m.sh.toFixed(3);
    $('dd-' + slug).textContent = m.mdd.toFixed(2) + '%';
    $('so-' + slug).textContent = m.so.toFixed(3);
  });

  const cols = [
    { key: 'fv',  label: 'Final Value',   isLower: false, fmt: v => '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits:2 }) },
    { key: 'tr',  label: 'Total Return',  isLower: false, fmt: v => fmtPct(v) },
    { key: 'sh',  label: 'Sharpe Ratio',  isLower: false, fmt: v => v.toFixed(3) },
    { key: 'so',  label: 'Sortino Ratio', isLower: false, fmt: v => v.toFixed(3) },
    { key: 'mdd', label: 'Max Drawdown',  isLower: true,  fmt: v => v.toFixed(2) + '%' },
  ];

  const best = {}, worst = {};
  cols.forEach(col => {
    const vals = AGENT_KEYS.map(n => metaMetrics[n] && metaMetrics[n][col.key]).filter(v => v != null);
    if (!vals.length) return;
    best[col.key]  = col.isLower ? Math.min(...vals) : Math.max(...vals);
    worst[col.key] = col.isLower ? Math.max(...vals) : Math.min(...vals);
  });

  const thead = '<tr><th>Agent</th>' + cols.map(c => '<th>' + c.label + '</th>').join('') + '</tr>';
  const tbody = AGENT_KEYS.map(name => {
    const m = metaMetrics[name];
    if (!m) return '';
    const cfg = AGENTS_CFG[name];
    const cells = cols.map(col => {
      const v = m[col.key];
      const isBest  = v === best[col.key];
      const isWorst = v === worst[col.key];
      const cls = isBest ? 'best' : (isWorst ? 'worst' : '');
      const prefix = isBest ? '▲ ' : (isWorst ? '▽ ' : '');
      return '<td class="' + cls + '">' + prefix + col.fmt(v) + '</td>';
    }).join('');
    return '<tr>' +
      '<td><div class="agent-cell">' +
        '<div class="aswatch" style="background:' + cfg.color + ';box-shadow:0 0 6px ' + cfg.color + '"></div>' +
        '<span class="acell-name">' + name + '</span>' +
      '</div></td>' +
      cells +
      '</tr>';
  }).join('');

  $('metricsTable').innerHTML = '<thead>' + thead + '</thead><tbody>' + tbody + '</tbody>';
}

// ── SSE control ───────────────────────────────────────────────────────────
function handlePlay() {
  if (isRunning) { stopSim(); return; }
  if (isDone)    { resetSim(); return; }
  startSim();
}

function startSim() {
  isRunning = true; isDone = false;
  const btn = $('playBtn');
  btn.className = 'play-btn stop';
  $('playIcon').textContent  = '■';
  $('playLabel').textContent = 'STOP';
  $('liveDot').classList.remove('idle');
  $('liveLabel').classList.remove('idle');
  $('liveLabel').textContent = 'LIVE';

  es = new EventSource('/api/run?delay=' + currentDelay);
  es.onmessage = function(e) {
    const data = JSON.parse(e.data);
    if (data.error)  { console.error(data.error); finishSim(); return; }
    if (data.init)   { metaMetrics = data.meta || {}; totalSteps = data.n || 0; return; }
    if (data.done)   { finishSim(); return; }
    applyStep(data);
  };
  es.onerror = function() { if (isRunning) finishSim(); };
}

function stopSim() {
  if (es) { es.close(); es = null; }
  isRunning = false;
  const btn = $('playBtn');
  btn.className = 'play-btn';
  $('playIcon').textContent  = '▶';
  $('playLabel').textContent = 'RESUME';
  $('liveDot').classList.add('idle');
  $('liveLabel').classList.add('idle');
  $('liveLabel').textContent = 'PAUSED';
}

function finishSim() {
  if (es) { es.close(); es = null; }
  isRunning = false; isDone = true;
  const btn = $('playBtn');
  btn.className = 'play-btn replay';
  $('playIcon').textContent  = '↺';
  $('playLabel').textContent = 'REPLAY';
  $('liveDot').classList.add('idle');
  $('liveLabel').classList.add('idle');
  $('liveLabel').textContent = 'DONE';
  $('progressFill').style.width = '100%';
  $('progressPct').textContent  = '100%';
  showMetrics();
}

function resetSim() {
  isDone = false; currentStep = 0;
  $('progressFill').style.width = '0%';
  $('progressPct').textContent  = '0%';
  $('currentDate').textContent  = '──────────';
  $('stepDisplay').textContent  = '0 / 0';
  $('regimeBadge').className    = 'regime-badge neutral';
  $('regimeBadge').textContent  = '● NEUTRAL';
  $('metricsSection').classList.add('hidden');

  [portfolioChart, drawdownChart, returnChart].forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(d => { d.data = []; });
    c.update('none');
  });
  allocChart.data.datasets.forEach(d => {
    d.data = new Array(ASSET_NAMES.length).fill(1 / ASSET_NAMES.length);
  });
  allocChart.update();

  AGENT_KEYS.forEach(name => {
    const slug = name.replace(/\s+/g, '_');
    $('val-' + slug).textContent = '$100,000.00';
    $('ret-' + slug).textContent = '+0.00%';
    $('ret-' + slug).className   = 'card-return zero';
    $('sh-'  + slug).textContent = '—';
    $('dd-'  + slug).textContent = '—';
    $('so-'  + slug).textContent = '—';
    const sp = sparkCharts[name];
    if (sp) { sp.data.labels = ['']; sp.data.datasets[0].data = [100000]; sp.update('none'); }
  });

  const btn = $('playBtn');
  btn.className = 'play-btn';
  $('playIcon').textContent  = '▶';
  $('playLabel').textContent = 'START SIMULATION';
  startSim();
}

function setSpeed(el) {
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  currentDelay = parseFloat(el.dataset.delay);
  if (isRunning) { stopSim(); startSim(); }
}

// ── Boot ──────────────────────────────────────────────────────────────────
buildCards();
buildLegends();
initCharts();
</script>
</body>
</html>"""
