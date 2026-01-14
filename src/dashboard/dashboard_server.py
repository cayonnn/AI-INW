# src/dashboard/dashboard_server.py
"""
🏦 HEDGE FUND GRADE Trading Dashboard (No External Dependencies)
=================================================================
Competition-Grade Professional Dashboard with Pure CSS/Canvas Charts
No CDN required - works offline!

Usage:
    python -m src.dashboard.dashboard_server
    Then open: http://localhost:8000
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from collections import deque
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from src.config.trading_profiles import get_active_profile
    HAS_TRADING = True
except:
    HAS_TRADING = False

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except:
    HAS_MT5 = False
    mt5 = None


def init_mt5():
    if not HAS_MT5: return False
    try: return mt5.initialize()
    except: return False


def get_mt5_positions(symbol=None):
    if not HAS_MT5 or not init_mt5(): return []
    try:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions: return []
        result = []
        for p in positions:
            pip = 0.01 if "XAU" in p.symbol else 0.0001
            pips = (p.price_current - p.price_open) / pip
            if p.type == 1: pips = -pips
            result.append({
                "ticket": p.ticket, "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume, "open_price": round(p.price_open, 2),
                "current_price": round(p.price_current, 2),
                "sl": round(p.sl, 2) if p.sl else 0,
                "tp": round(p.tp, 2) if p.tp else 0,
                "profit": round(p.profit, 2), "pips": round(pips, 1),
                "duration": str(datetime.now() - datetime.fromtimestamp(p.time)).split('.')[0],
            })
        return result
    except: return []


def get_mt5_account():
    if not HAS_MT5 or not init_mt5(): return {}
    try:
        i = mt5.account_info()
        if not i: return {}
        dd = ((i.balance - i.equity) / i.balance * 100) if i.balance > 0 else 0
        return {
            "balance": round(i.balance, 2), "equity": round(i.equity, 2),
            "margin": round(i.margin, 2), "margin_free": round(i.margin_free, 2),
            "margin_level": round(i.margin_level, 2) if i.margin_level else 0,
            "profit": round(i.profit, 2), "leverage": i.leverage,
            "currency": i.currency, "server": i.server, "drawdown_pct": round(max(0, dd), 2),
        }
    except: return {}


def get_trade_history(days=30):
    """Get closed trade history from MT5 deals - only exit deals with profit."""
    if not HAS_MT5 or not init_mt5(): return []
    try:
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now() + timedelta(days=1)
        
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals: return []
        
        result = []
        for d in deals:
            # Only show EXIT deals (entry=1) - these have the actual profit
            if d.type > 1:  # Skip balance operations
                continue
            if d.entry != 1:  # Skip entry deals, only show exit (closed) deals
                continue
            
            # For EXIT deals, the type is the CLOSING action:
            # - SELL deal (type=1) = closing a BUY position → original was BUY
            # - BUY deal (type=0) = closing a SELL position → original was SELL
            # So we INVERT the type to show the original position type
            original_type = "BUY" if d.type == 1 else "SELL"
                
            result.append({
                "ticket": d.ticket,
                "order": d.order,
                "symbol": d.symbol,
                "type": original_type,  # Show original position type, not closing action
                "volume": d.volume,
                "price": round(d.price, 2),
                "profit": round(d.profit + d.swap + d.commission, 2),
                "time": datetime.fromtimestamp(d.time).isoformat(),
            })
        
        # Sort by time, newest first
        return sorted(result, key=lambda x: x["time"], reverse=True)
    except Exception as e:
        print(f"History error: {e}")
        return []


def get_statistics(days=30):
    if not HAS_MT5 or not init_mt5(): return _empty_stats()
    try:
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=days), datetime.now() + timedelta(days=1))
        if not deals: return _empty_stats()
        closes = [d for d in deals if d.entry == 1 and d.type in [0, 1]]
        if not closes: return _empty_stats()
        profits = [d.profit for d in closes]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        tw, tl = sum(wins) if wins else 0, abs(sum(losses)) if losses else 0
        daily = {}
        for d in closes:
            day = datetime.fromtimestamp(d.time).date()
            daily[day] = daily.get(day, 0) + d.profit
        rets = list(daily.values())
        sharpe = _sharpe(rets)
        max_dd = _max_dd(closes)
        return {
            "total_trades": len(closes), "winning_trades": len(wins), "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(closes) * 100, 1) if closes else 0,
            "total_profit": round(tw, 2), "total_loss": round(tl, 2), "net_profit": round(tw - tl, 2),
            "best_trade": round(max(profits), 2) if profits else 0,
            "worst_trade": round(min(profits), 2) if profits else 0,
            "average_profit": round(tw / len(wins), 2) if wins else 0,
            "average_loss": round(tl / len(losses), 2) if losses else 0,
            "profit_factor": round(tw / tl, 2) if tl > 0 else 0,
            "sharpe_ratio": round(sharpe, 2), "max_drawdown": round(max_dd, 2),
        }
    except: return _empty_stats()


def _empty_stats():
    return {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate": 0,
            "total_profit": 0, "total_loss": 0, "net_profit": 0, "best_trade": 0, "worst_trade": 0,
            "average_profit": 0, "average_loss": 0, "profit_factor": 0, "sharpe_ratio": 0, "max_drawdown": 0}


def _sharpe(rets):
    if len(rets) < 2: return 0
    avg = sum(rets) / len(rets)
    std = math.sqrt(sum((r - avg) ** 2 for r in rets) / len(rets))
    return (avg / std * math.sqrt(252)) if std > 0 else 0


def _max_dd(trades):
    if not trades: return 0
    cum, peak, dd = 0, 0, 0
    for t in sorted(trades, key=lambda x: x.time):
        cum += t.profit
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def get_equity_curve(days=7):
    if not HAS_MT5 or not init_mt5(): return []
    try:
        acc = mt5.account_info()
        if not acc: return []
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=days), datetime.now() + timedelta(days=1))
        if not deals: return [{"time": datetime.now().isoformat(), "equity": acc.balance}]
        sorted_d = sorted(deals, key=lambda x: x.time)
        bal = acc.balance
        for d in sorted_d:
            if d.profit != 0: bal -= d.profit
        curve = []
        for d in sorted_d:
            if d.profit != 0:
                bal += d.profit
                curve.append({"time": datetime.fromtimestamp(d.time).isoformat(), "equity": round(bal, 2)})
        return curve if curve else [{"time": datetime.now().isoformat(), "equity": acc.balance}]
    except: return []


def get_live_price(symbol="XAUUSD"):
    """Get live price for symbol."""
    if not HAS_MT5 or not init_mt5(): return {}
    try:
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return {}
        info = mt5.symbol_info(symbol)
        spread = round((tick.ask - tick.bid) / info.point, 1) if info else 0
        return {
            "symbol": symbol,
            "bid": round(tick.bid, 2),
            "ask": round(tick.ask, 2),
            "spread": spread,
            "time": datetime.fromtimestamp(tick.time).isoformat(),
        }
    except: return {}


def get_daily_pnl(days=30):
    """Get daily P/L for calendar heatmap."""
    if not HAS_MT5 or not init_mt5(): return []
    try:
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=days), datetime.now() + timedelta(days=1))
        if not deals: return []
        daily = {}
        for d in deals:
            if d.entry == 1 and d.type in [0, 1]:
                day = datetime.fromtimestamp(d.time).strftime("%Y-%m-%d")
                daily[day] = daily.get(day, 0) + d.profit
        return [{"date": k, "pnl": round(v, 2)} for k, v in sorted(daily.items())]
    except: return []


def get_account_growth():
    """Calculate account growth percentage."""
    if not HAS_MT5 or not init_mt5(): return {}
    try:
        acc = mt5.account_info()
        if not acc: return {}
        # Get 30-day history
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now() + timedelta(days=1))
        if not deals:
            return {"current": acc.balance, "previous": acc.balance, "growth_pct": 0, "growth_amt": 0}
        
        # Calculate starting balance 30 days ago
        total_pnl = sum(d.profit for d in deals if d.profit != 0)
        prev_balance = acc.balance - total_pnl
        
        if prev_balance <= 0:
            prev_balance = acc.balance
        
        growth_pct = ((acc.balance - prev_balance) / prev_balance * 100) if prev_balance > 0 else 0
        
        return {
            "current": round(acc.balance, 2),
            "previous": round(prev_balance, 2),
            "growth_pct": round(growth_pct, 2),
            "growth_amt": round(acc.balance - prev_balance, 2),
        }
    except: return {}


class LogBuffer:
    def __init__(self): self._buf = deque(maxlen=200)
    def add(self, lvl, msg): self._buf.append({"ts": datetime.now().isoformat(), "lvl": lvl, "msg": msg})
    def get(self, n=50): return list(self._buf)[-n:]
    def clear(self): self._buf.clear()


log_buffer = LogBuffer()


class DashboardLogHandler(logging.Handler):
    def emit(self, r):
        try: log_buffer.add(r.levelname, self.format(r))
        except: pass


if HAS_FASTAPI:
    app = FastAPI(title="Hedge Fund Dashboard", version="5.0")
    connections = []

    HTML = '''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🏦 Hedge Fund Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a12;--card:#141420;--border:#252535;--text:#fff;--dim:#666;--green:#00d9a5;--red:#ff4d6a;--blue:#4a9eff;--purple:#a855f7;--gold:linear-gradient(135deg,#ffd700,#ff6b35)}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.header{background:rgba(20,20,32,.9);backdrop-filter:blur(10px);border-bottom:1px solid var(--border);padding:16px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:99}
.logo{font-size:1.3rem;font-weight:700;background:var(--gold);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.live{display:flex;align-items:center;gap:8px;color:var(--green);font-size:.85rem}
.live::before{content:'';width:8px;height:8px;background:var(--green);border-radius:50%;animation:blink 2s infinite}
@keyframes blink{50%{opacity:.3}}
.container{max-width:1600px;margin:0 auto;padding:20px}
.grid{display:grid;gap:16px;margin-bottom:20px}
.g4{grid-template-columns:repeat(4,1fr)}
.g3{grid-template-columns:repeat(3,1fr)}
.g2{grid-template-columns:repeat(2,1fr)}
@media(max-width:1200px){.g4,.g3{grid-template-columns:repeat(2,1fr)}}
@media(max-width:768px){.g4,.g3,.g2{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;transition:all .2s}
.card:hover{border-color:var(--blue);transform:translateY(-2px)}
.card-title{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.card-value{font-size:1.6rem;font-weight:700}
.card-sub{font-size:.75rem;color:var(--dim);margin-top:4px}
.profit{color:var(--green)}.loss{color:var(--red)}.neutral{color:var(--blue)}
.bar{height:6px;background:#1a1a2a;border-radius:3px;margin-top:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .5s}
.ring-box{display:flex;justify-content:center;padding:16px}
.ring{width:90px;height:90px;border-radius:50%;background:conic-gradient(var(--green) calc(var(--r)*1%),var(--red) 0);display:flex;align-items:center;justify-content:center;position:relative}
.ring::before{content:'';position:absolute;width:60px;height:60px;background:var(--card);border-radius:50%}
.ring span{position:relative;font-weight:700}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.stat{background:#1a1a28;border-radius:8px;padding:12px;text-align:center}
.stat-val{font-size:1.2rem;font-weight:700}
.stat-lbl{font-size:.65rem;color:var(--dim);margin-top:2px}
.chart-container{position:relative;height:220px;padding:10px 0}
.chart-svg{width:100%;height:100%}
.chart-grid{stroke:#252535;stroke-width:1}
.chart-line{fill:none;stroke:var(--blue);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.chart-area{fill:url(#gradient);opacity:.3}
.chart-dot{fill:var(--blue);cursor:pointer;transition:r .2s}
.chart-dot:hover{r:6}
.chart-label{fill:var(--dim);font-size:10px}
.chart-tooltip{position:absolute;background:#1a1a28;border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:.75rem;pointer-events:none;opacity:0;transition:opacity .2s}
.chart-tooltip.show{opacity:1}
table{width:100%;border-collapse:collapse}
th,td{padding:10px;text-align:left;border-bottom:1px solid var(--border);font-size:.8rem}
th{color:var(--dim);font-size:.65rem;text-transform:uppercase}
tr:hover{background:#1a1a28}
.badge{padding:3px 8px;border-radius:12px;font-size:.7rem;font-weight:600}
.badge-buy{background:rgba(0,217,165,.15);color:var(--green)}
.badge-sell{background:rgba(255,77,106,.15);color:var(--red)}
.logs{height:180px;background:#000;border-radius:8px;padding:10px;overflow-y:auto;font-family:monospace;font-size:.7rem}
.log{padding:2px 0;border-bottom:1px solid #111}
.log-t{color:#444}.log-I{color:var(--blue)}.log-W{color:orange}.log-E{color:var(--red)}
.ticker{display:flex;align-items:center;gap:20px;background:#1a1a28;padding:8px 16px;border-radius:8px;font-size:.85rem}
.ticker-price{font-size:1.2rem;font-weight:700;transition:color .3s}
.ticker-up{color:var(--green);animation:flash-green .5s}
.ticker-down{color:var(--red);animation:flash-red .5s}
@keyframes flash-green{0%{background:rgba(0,217,165,.3)}100%{background:transparent}}
@keyframes flash-red{0%{background:rgba(255,77,106,.3)}100%{background:transparent}}
.spread{color:var(--dim);font-size:.75rem}
.growth{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:12px;font-size:.8rem;font-weight:600}
.growth-up{background:rgba(0,217,165,.15);color:var(--green)}
.growth-down{background:rgba(255,77,106,.15);color:var(--red)}
.calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;margin-top:12px}
.cal-day{aspect-ratio:1;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:.6rem;cursor:pointer;transition:transform .2s}
.cal-day:hover{transform:scale(1.15)}
.cal-profit{background:var(--green)}.cal-loss{background:var(--red)}.cal-zero{background:#2a2a3a}
</style>
</head>
<body>
<div class="header">
<span class="logo">🏦 แดชบอร์ดกองทุน</span>
<div class="ticker" id="ticker">
<span>🥇 XAUUSD</span>
<span class="ticker-price" id="price">0.00</span>
<span class="spread">Spread: <span id="spread">0</span></span>
</div>
<div style="display:flex;align-items:center;gap:12px">
<span class="growth" id="growth">📈 0%</span>
<div class="live"><span id="srv">กำลังเชื่อมต่อ...</span></div>
</div>
</div>
<div class="container">
<div class="grid g4">
<div class="card"><div class="card-title">💰 ยอดเงิน</div><div class="card-value neutral" id="bal">$0</div><div class="card-sub" id="cur">USD</div></div>
<div class="card"><div class="card-title">📊 สินทรัพย์</div><div class="card-value neutral" id="eq">$0</div><div class="card-sub">มาร์จิน: <span id="ml">0%</span></div></div>
<div class="card"><div class="card-title">📈 กำไร/ขาดทุนลอย</div><div class="card-value" id="fl">$0</div><div class="card-sub"><span id="pc">0</span> ออเดอร์</div></div>
<div class="card"><div class="card-title">🎯 กำไรสุทธิ (30วัน)</div><div class="card-value" id="net">$0</div><div class="card-sub"><span id="tc">0</span> เทรด</div></div>
</div>
<div class="grid g4">
<div class="card"><div class="card-title">📐 อัตราชาร์ป</div><div class="card-value neutral" id="sr">0</div><div class="bar"><div class="bar-fill" id="sr-b" style="width:0;background:var(--blue)"></div></div></div>
<div class="card"><div class="card-title">📉 ขาดทุนสูงสุด</div><div class="card-value loss" id="dd">$0</div><div class="bar"><div class="bar-fill" id="dd-b" style="width:0;background:var(--red)"></div></div></div>
<div class="card"><div class="card-title">⚖️ อัตรากำไร</div><div class="card-value profit" id="pf">0</div><div class="bar"><div class="bar-fill" id="pf-b" style="width:0;background:var(--green)"></div></div></div>
<div class="card"><div class="card-title">🎲 ค่าคาดหวัง</div><div class="card-value neutral" id="ex">$0</div><div class="bar"><div class="bar-fill" id="ex-b" style="width:0;background:var(--purple)"></div></div></div>
</div>
<div class="grid g3">
<div class="card">
<div class="card-title">🏆 ผลการเทรด</div>
<div class="stats">
<div class="stat"><div class="stat-val profit" id="bst">$0</div><div class="stat-lbl">กำไรสูงสุด</div></div>
<div class="stat"><div class="stat-val loss" id="wst">$0</div><div class="stat-lbl">ขาดทุนสูงสุด</div></div>
<div class="stat"><div class="stat-val profit" id="aw">$0</div><div class="stat-lbl">เฉลี่ยกำไร</div></div>
<div class="stat"><div class="stat-val loss" id="al">$0</div><div class="stat-lbl">เฉลี่ยขาดทุน</div></div>
</div>
<div class="ring-box"><div class="ring" id="ring" style="--r:0"><span id="wr">0%</span></div></div>
<div style="text-align:center;font-size:.75rem;color:var(--dim)"><span class="profit" id="wc">0</span> ชนะ / <span class="loss" id="lc">0</span> แพ้</div>
</div>
<div class="card" style="grid-column:span 2">
<div class="card-title">📈 กราฟสินทรัพย์ (7 วัน)</div>
<div class="chart-container">
<svg class="chart-svg" id="chart" viewBox="0 0 600 200" preserveAspectRatio="none">
<defs><linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#4a9eff" stop-opacity="0.4"/><stop offset="100%" stop-color="#4a9eff" stop-opacity="0"/></linearGradient></defs>
<g id="chart-grid"></g><path id="chart-area" class="chart-area"/><path id="chart-line" class="chart-line"/><g id="chart-dots"></g><g id="chart-labels"></g>
</svg>
<div class="chart-tooltip" id="tooltip"></div>
</div>
</div>
</div>
<div class="grid g2">
<div class="card">
<div class="card-title">💼 ออเดอร์ที่เปิดอยู่</div>
<div style="max-height:280px;overflow-y:auto">
<table><thead><tr><th>ตั๋ว</th><th>สัญลักษณ์</th><th>ประเภท</th><th>Lot</th><th>ราคาเข้า</th><th>ราคาปัจจุบัน</th><th>Pips</th><th>กำไร/ขาดทุน</th></tr></thead>
<tbody id="pos"><tr><td colspan="8" style="text-align:center;color:var(--dim)">ไม่มีออเดอร์</td></tr></tbody></table>
</div>
</div>
<div class="card">
<div class="card-title">📅 ปฏิทินกำไร/ขาดทุน (14 วัน)</div>
<div class="calendar-header" style="display:grid;grid-template-columns:repeat(7,1fr);gap:3px;margin-bottom:8px;font-size:.65rem;color:var(--dim);text-align:center">
<span>อา</span><span>จ</span><span>อ</span><span>พ</span><span>พฤ</span><span>ศ</span><span>ส</span>
</div>
<div class="calendar" id="calendar"></div>
<div style="display:flex;justify-content:space-between;margin-top:10px;font-size:.7rem;color:var(--dim)">
<span>🔴 ขาดทุน</span><span>⚪ ไม่มีเทรด</span><span>🟢 กำไร</span>
</div>
</div>
</div>
<div class="grid g2">
<div class="card">
<div class="card-title">📜 ประวัติการเทรด</div>
<div style="max-height:280px;overflow-y:auto">
<table><thead><tr><th>เวลา</th><th>สัญลักษณ์</th><th>ประเภท</th><th>Lot</th><th>กำไร/ขาดทุน</th></tr></thead>
<tbody id="hist"><tr><td colspan="5" style="text-align:center;color:var(--dim)">ไม่มีประวัติ</td></tr></tbody></table>
</div>
</div>
<div class="card">
<div class="card-title">📋 บันทึกระบบ</div>
<div class="logs" id="logs"><div class="log"><span class="log-t">--:--</span> กำลังเชื่อมต่อ...</div></div>
</div>
</div>
</div>
<script>
function fmt(n){return (n>=0?'+$':'-$')+Math.abs(n).toFixed(2)}
function upd(d){
if(d.account){
document.getElementById('bal').textContent='$'+d.account.balance.toFixed(2);
document.getElementById('eq').textContent='$'+d.account.equity.toFixed(2);
document.getElementById('ml').textContent=(d.account.margin_level||0).toFixed(0)+'%';
document.getElementById('cur').textContent=d.account.currency||'USD';
document.getElementById('srv').textContent=d.account.server||'Connected';
const f=document.getElementById('fl');
f.textContent=fmt(d.account.profit||0);
f.className='card-value '+(d.account.profit>=0?'profit':'loss');
}
if(d.positions){
document.getElementById('pc').textContent=d.positions.length;
const t=document.getElementById('pos');
t.innerHTML=d.positions.length?d.positions.map(p=>`<tr><td>${p.ticket}</td><td>${p.symbol}</td><td><span class="badge badge-${p.type.toLowerCase()}">${p.type}</span></td><td>${p.volume}</td><td>${p.open_price}</td><td>${p.current_price}</td><td class="${p.pips>=0?'profit':'loss'}">${p.pips>0?'+':''}${p.pips}</td><td class="${p.profit>=0?'profit':'loss'}">${fmt(p.profit)}</td></tr>`).join(''):'<tr><td colspan="8" style="text-align:center;color:var(--dim)">No positions</td></tr>';
}
if(d.statistics){const s=d.statistics;
document.getElementById('net').textContent=fmt(s.net_profit);
document.getElementById('net').className='card-value '+(s.net_profit>=0?'profit':'loss');
document.getElementById('tc').textContent=s.total_trades;
document.getElementById('sr').textContent=s.sharpe_ratio.toFixed(2);
document.getElementById('sr-b').style.width=Math.min(100,Math.abs(s.sharpe_ratio)*25)+'%';
document.getElementById('dd').textContent=fmt(-s.max_drawdown);
document.getElementById('dd-b').style.width=Math.min(100,s.max_drawdown/100)+'%';
document.getElementById('pf').textContent=s.profit_factor.toFixed(2);
document.getElementById('pf-b').style.width=Math.min(100,s.profit_factor*20)+'%';
const exp=(s.win_rate/100*s.average_profit)-(1-s.win_rate/100)*s.average_loss;
document.getElementById('ex').textContent=fmt(exp);
document.getElementById('ex-b').style.width=Math.min(100,Math.abs(exp))+'%';
document.getElementById('bst').textContent=fmt(s.best_trade);
document.getElementById('wst').textContent=fmt(s.worst_trade);
document.getElementById('aw').textContent=fmt(s.average_profit);
document.getElementById('al').textContent=fmt(-s.average_loss);
document.getElementById('wr').textContent=s.win_rate+'%';
document.getElementById('ring').style.setProperty('--r',s.win_rate);
document.getElementById('wc').textContent=s.winning_trades;
document.getElementById('lc').textContent=s.losing_trades;
}
if(d.equity_curve&&d.equity_curve.length>1){
const data=d.equity_curve;
const vals=data.map(e=>e.equity);
const min=Math.min(...vals),max=Math.max(...vals);
const range=max-min||1;
const w=600,h=180,padX=50,padY=20;
const stepX=(w-padX)/(data.length-1);
const scaleY=v=>(h-padY)-((v-min)/range)*(h-padY-10);
let pts=data.map((e,i)=>[padX+i*stepX,scaleY(e.equity)]);
let lineD='M'+pts.map(p=>p.join(',')).join('L');
let areaD=lineD+'L'+(padX+(data.length-1)*stepX)+','+(h-padY)+'L'+padX+','+(h-padY)+'Z';
document.getElementById('chart-line').setAttribute('d',lineD);
document.getElementById('chart-area').setAttribute('d',areaD);
let grid='';for(let i=0;i<5;i++){const y=padY+i*(h-padY-10)/4;grid+=`<line x1="${padX}" y1="${y}" x2="${w}" y2="${y}" class="chart-grid"/>`;}document.getElementById('chart-grid').innerHTML=grid;
let dots='',labels='';
const step=Math.max(1,Math.floor(data.length/6));
for(let i=0;i<data.length;i++){
dots+=`<circle cx="${pts[i][0]}" cy="${pts[i][1]}" r="3" class="chart-dot" data-val="$${data[i].equity.toFixed(2)}" data-date="${new Date(data[i].time).toLocaleDateString()}"/>`;
if(i%step===0||i===data.length-1)labels+=`<text x="${pts[i][0]}" y="${h-5}" class="chart-label" text-anchor="middle">${new Date(data[i].time).toLocaleDateString('en',{month:'short',day:'numeric'})}</text>`;
}
for(let i=0;i<5;i++){const y=padY+i*(h-padY-10)/4;const v=max-i*(range/4);labels+=`<text x="${padX-5}" y="${y+3}" class="chart-label" text-anchor="end">$${v.toFixed(0)}</text>`;}
document.getElementById('chart-dots').innerHTML=dots;
document.getElementById('chart-labels').innerHTML=labels;
document.querySelectorAll('.chart-dot').forEach(dot=>{dot.onmouseenter=e=>{const t=document.getElementById('tooltip');t.innerHTML=`<b>${e.target.dataset.date}</b><br>${e.target.dataset.val}`;t.style.left=e.offsetX+'px';t.style.top=(e.offsetY-50)+'px';t.classList.add('show');};dot.onmouseleave=()=>document.getElementById('tooltip').classList.remove('show');});
}
if(d.history&&d.history.length>0){
const c=d.history.slice(0,15);
document.getElementById('hist').innerHTML=c.map(t=>`<tr><td>${new Date(t.time).toLocaleString('th-TH')}</td><td>${t.symbol}</td><td><span class="badge badge-${t.type.toLowerCase()}">${t.type}</span></td><td>${t.volume}</td><td class="${t.profit>=0?'profit':'loss'}">${fmt(t.profit)}</td></tr>`).join('');
}else{document.getElementById('hist').innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--dim)">ไม่มีประวัติ</td></tr>';}
if(d.logs){d.logs.forEach(l=>{
const b=document.getElementById('logs');
const e=document.createElement('div');e.className='log';
e.innerHTML=`<span class="log-t">${new Date(l.ts).toLocaleTimeString()}</span> <span class="log-${l.lvl[0]}">[${l.lvl}]</span> ${l.msg}`;
b.appendChild(e);b.scrollTop=b.scrollHeight;
while(b.children.length>40)b.removeChild(b.firstChild);
});}
if(d.price){
const p=document.getElementById('price');
const newPrice=d.price.bid;
const oldPrice=parseFloat(p.textContent)||0;
p.textContent=newPrice.toFixed(2);
p.classList.remove('ticker-up','ticker-down');
if(newPrice>oldPrice)p.classList.add('ticker-up');
else if(newPrice<oldPrice)p.classList.add('ticker-down');
document.getElementById('spread').textContent=d.price.spread;
}
if(d.growth){
const g=document.getElementById('growth');
const pct=d.growth.growth_pct||0;
g.textContent=(pct>=0?'↑':'↓')+Math.abs(pct).toFixed(1)+'%';
g.className='growth '+(pct>=0?'growth-up':'growth-down');
}
if(d.daily_pnl){
const cal=document.getElementById('calendar');
const today=new Date();
const startDate=new Date(today);
startDate.setDate(startDate.getDate()-13);
const firstDay=startDate.getDay();
let html='';
for(let i=0;i<firstDay;i++){html+='<div class="cal-day" style="opacity:0.2"></div>';}
const pnlMap={};
d.daily_pnl.forEach(p=>pnlMap[p.date]=p.pnl);
for(let i=0;i<14;i++){
const date=new Date(startDate);
date.setDate(date.getDate()+i);
const key=date.toISOString().split('T')[0];
const pnl=pnlMap[key]||0;
const day=date.getDate();
let cls='cal-zero',title=key+': ไม่มีเทรด';
if(pnl>0){cls='cal-profit';title=key+': +$'+pnl.toFixed(2);}
else if(pnl<0){cls='cal-loss';title=key+': -$'+Math.abs(pnl).toFixed(2);}
const intensity=Math.min(1,Math.abs(pnl)/100);
const opacity=0.3+intensity*0.7;
html+=`<div class="cal-day ${cls}" style="opacity:${opacity}" title="${title}">${day}</div>`;
}
cal.innerHTML=html;
}
}
async function load(){try{const r=await fetch('/api/status');upd(await r.json());}catch(e){}}
let ws;function conn(){
ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=e=>upd(JSON.parse(e.data));
ws.onclose=()=>setTimeout(conn,3000);
}
conn();load();setInterval(load,3000);
</script>
</body></html>'''

    @app.get("/", response_class=HTMLResponse)
    async def index(): return HTML

    @app.get("/api/status")
    async def status():
        return {
            "account": get_mt5_account(),
            "positions": get_mt5_positions("XAUUSD"),
            "statistics": get_statistics(30),
            "history": get_trade_history(7),
            "equity_curve": get_equity_curve(7),
            "price": get_live_price("XAUUSD"),
            "growth": get_account_growth(),
            "daily_pnl": get_daily_pnl(14),
            "logs": log_buffer.get(10),
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        import asyncio
        await websocket.accept()
        connections.append(websocket)
        try:
            while True:
                await asyncio.sleep(2)
                try:
                    await websocket.send_json(await status())
                except: break
        except: pass
        finally:
            if websocket in connections: connections.remove(websocket)

    @app.get("/api/risk/stack")
    async def risk_stack_api():
        """Get current effective risk stack."""
        try:
            from src.analytics.risk_stack import get_risk_stack_snapshot
            snapshot = get_risk_stack_snapshot()
            return snapshot if snapshot else {"error": "No risk stack data yet"}
        except Exception as e:
            return {"error": str(e)}


def run_dashboard(host="0.0.0.0", port=8000):
    if not HAS_FASTAPI:
        print("pip install fastapi uvicorn")
        return
    h = DashboardLogHandler()
    h.setLevel(logging.INFO)
    logging.getLogger().addHandler(h)
    
    # Add startup logs
    log_buffer.add("INFO", "🏦 Dashboard started")
    log_buffer.add("INFO", f"MT5 Connected: {HAS_MT5 and init_mt5()}")
    
    acc = get_mt5_account()
    if acc:
        log_buffer.add("INFO", f"Account: {acc.get('server', 'N/A')}")
        log_buffer.add("INFO", f"Balance: ${acc.get('balance', 0):.2f}")
        log_buffer.add("INFO", f"Equity: ${acc.get('equity', 0):.2f}")
        
        pos = get_mt5_positions()
        if pos:
            log_buffer.add("INFO", f"Open positions: {len(pos)}")
            total_pl = sum(p['profit'] for p in pos)
            log_buffer.add("INFO", f"Floating P/L: ${total_pl:.2f}")
    
    print(f"🏦 Dashboard: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    run_dashboard(a.host, a.port)
