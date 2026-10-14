"""brooks_core.py — Al Brooks price action, mechanised (specification "BPA-1.0", frozen 2026-10-09).

Everything here is causal: a value at bar t uses bars <= t only; k=2 pivots become visible at bar j+2.
Definitions follow the public Brooks glossary/articles as formalised in the research specification (sections A1-A12);
every threshold is ATR- or ABR-relative and listed in DEFAULTS so it can be pre-registered and neighbour-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# --------------------------------------------------------------------------- parameters (defaults = spec bold values)
DEFAULTS = dict(
    doji=0.25, strong_br=0.60, strong_cpos=0.75, strong_r=1.0, surprise_r=2.5, sig_min=4, sig_overlap=0.67,
    zz_atr=1.0, tl_break=0.25, ema_test=0.25, dt_tol=0.25, wedge_tol=0.10, wedge_min=8, wedge_max=60,
    spike_run=2, spike_move=1.5, tr_stall=20, tr_min=20, tight_h=2.0, broad_h=2.0, bo_close=0.10, bo_fail_k=3,
    micro_ch=5, climax_ch=10, gap_run=20, h_tol=0.10, ai_conf=0.0, mtr_test=0.5, mtr_expire=60, ff_bars=8,
    ff_h=1.5, ff_age=30, second_entry=10, room=1, tstop_swing=48, tstop_scalp=12, sess_or=18,
    p_f1=0.6, p_f2=0.6, p_f3=0.6, p_f4=0.4, p_f5=0.6, p_f6=0.45,
    # BPA-1.1 (fidelity fixes from the chart audit)
    no_late=4,          # US session: no new entries when fewer than this many bars remain (last 20 minutes)
    f4_window=15,       # F4 second entry: first setup must trigger, fail and be re-tested within this many bars
    f4_retest=0.25,     # ... the re-test reaches the first setup's low within this many ATR
    spike_block=2,      # F2: no fade when the last 5 bars hold >= this many strong bars toward the edge
    or_move=0.5,        # F6: opening reversal needs a move of >= this many ATR from the open into the magnet
)
PKEYS = list(DEFAULTS)
_K = {k: i for i, k in enumerate(PKEYS)}
K_SIG, K_ZZ, K_TLB, K_EMAT, K_DT, K_WT = _K["sig_min"], _K["zz_atr"], _K["tl_break"], _K["ema_test"], _K["dt_tol"], _K["wedge_tol"]
K_WMIN, K_WMAX, K_SRUN, K_SMOVE, K_STALL, K_TRMIN = _K["wedge_min"], _K["wedge_max"], _K["spike_run"], _K["spike_move"], _K["tr_stall"], _K["tr_min"]
K_TIGHT, K_BROAD, K_BOC, K_BOK, K_MC, K_CMC = _K["tight_h"], _K["broad_h"], _K["bo_close"], _K["bo_fail_k"], _K["micro_ch"], _K["climax_ch"]
K_GAP, K_HTOL, K_AIC, K_MTRT, K_MTRX, K_FFB = _K["gap_run"], _K["h_tol"], _K["ai_conf"], _K["mtr_test"], _K["mtr_expire"], _K["ff_bars"]
K_FFH, K_FFA, K_SEC, K_ROOM, K_TSW, K_TSC, K_SOR = _K["ff_h"], _K["ff_age"], _K["second_entry"], _K["room"], _K["tstop_swing"], _K["tstop_scalp"], _K["sess_or"]
K_P0 = _K["p_f1"]
K_LATE, K_F4W, K_F4R, K_SPB, K_ORM = _K["no_late"], _K["f4_window"], _K["f4_retest"], _K["spike_block"], _K["or_move"]
assert [PKEYS[K_P0 + i] for i in range(6)] == ["p_f1", "p_f2", "p_f3", "p_f4", "p_f5", "p_f6"]
FAM = ["F1_H2L2", "F2_TR_fade", "F3_failed_BO", "F4_reversal", "F5_breakout", "F6_opening"]
PRIORITY = (3, 2, 0, 4, 1, 5)          # F4 -> F3 -> F1 -> F5 -> F2 -> F6

# intent fields (IT[t, f, k])
I_SIDE, I_TYPE, I_EPX, I_STA, I_STB, I_TGA, I_TGB, I_FRA, I_TRB, I_TST, I_P0, I_TE, I_BE = range(13)
NI = 13
T_STOP, T_LIMIT, T_MARKET = 1, 2, 3

# feature matrix columns
(C_O, C_H, C_L, C_C, C_ATR, C_ABR, C_EMA, C_R, C_DOJI, C_BTB, C_STB, C_SB, C_SS, C_SURU, C_SURD, C_INS, C_OVL,
 C_SIGB, C_SIGS, C_CPOS, C_HH20, C_LL20, C_EMASL, C_SIDE20, C_PH2, C_PL2, C_TS, C_DAY, C_SBAR, C_PDH, C_PDL,
 C_DOP, C_PCL, C_SLAST, C_LEFT) = range(35)
NF = 35


def pvec(**over):
    d = dict(DEFAULTS)
    d.update(over)
    return np.array([float(d[k]) for k in PKEYS])


def pidx(name):
    return PKEYS.index(name)


# --------------------------------------------------------------------------- vectorised features
@njit(cache=True)
def _ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty(len(x))
    e = x[0]
    for i in range(len(x)):
        e = x[i] if i == 0 else a * x[i] + (1 - a) * e
        out[i] = e
    return out


@njit(cache=True)
def _rma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    v = np.mean(x[:n])
    out[n - 1] = v
    for i in range(n, len(x)):
        v = (v * (n - 1) + x[i]) / n
        out[i] = v
    return out


@njit(cache=True)
def _run_len(mask):
    out = np.zeros(len(mask))
    k = 0
    for i in range(len(mask)):
        k = k + 1 if mask[i] else 0
        out[i] = k
    return out


@njit(cache=True)
def _pivots(H, L, k):
    """pivot high at j: H[j] >= max of k bars before and after (uses future; consumers read it at j+k)"""
    n = len(H)
    ph = np.zeros(n, np.bool_)
    pl = np.zeros(n, np.bool_)
    for j in range(k, n - k):
        if H[j] >= H[j - k:j].max() and H[j] >= H[j + 1:j + k + 1].max():
            ph[j] = True
        if L[j] <= L[j - k:j].min() and L[j] <= L[j + 1:j + k + 1].min():
            pl[j] = True
    return ph, pl


def features(df: pd.DataFrame, P, session="crypto"):
    """df: 5m bars (UTC index; open/high/low/close). session: 'crypto' (UTC days, no session) or 'rth' (US equities)."""
    p = dict(zip(PKEYS, P))
    O, H, L, C = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    n = len(C)
    R = H - L
    Rs = np.where(R > 0, R, np.nan)
    br = np.where(R > 0, np.abs(C - O) / Rs, 0.0)
    cpos = np.where(R > 0, (C - L) / Rs, 0.5)
    prevC = np.r_[C[0], C[:-1]]
    TR = np.maximum(R, np.maximum(np.abs(H - prevC), np.abs(L - prevC)))
    TR[0] = R[0]
    atr = _rma(TR, 20)
    abr = pd.Series(R).rolling(20, min_periods=20).mean().to_numpy()
    ema = _ema(C, 20)
    doji = br < p["doji"]
    btb = (C > O) & ~doji
    stb = (C < O) & ~doji
    sb = btb & (br >= p["strong_br"]) & (cpos >= p["strong_cpos"]) & (R >= p["strong_r"] * abr)
    ss = stb & (br >= p["strong_br"]) & ((1 - cpos) >= p["strong_cpos"]) & (R >= p["strong_r"] * abr)
    suru = sb & (R >= p["surprise_r"] * abr)
    surd = ss & (R >= p["surprise_r"] * abr)
    Hp, Lp = np.r_[np.nan, H[:-1]], np.r_[np.nan, L[:-1]]
    ins = (H <= Hp) & (L >= Lp)
    inter = np.clip(np.minimum(H, Hp) - np.maximum(L, Lp), 0, None)
    ovl = np.where(R > 0, inter / Rs, 1.0)
    ovl[0] = 0.0
    lt = np.where(R > 0, (np.minimum(O, C) - L) / Rs, 0)
    ut = np.where(R > 0, (H - np.maximum(O, C)) / Rs, 0)
    sz = (R >= 0.5 * abr) & (R <= 2.0 * abr)
    sigb = np.where(cpos >= 0.5, (C > O).astype(int) + (cpos >= 0.75) + (lt >= 1 / 3) + (ut <= 0.25) + sz
                    + (ovl <= p["sig_overlap"]), 0)
    sigs = np.where(cpos <= 0.5, (C < O).astype(int) + (cpos <= 0.25) + (ut >= 1 / 3) + (lt <= 0.25) + sz
                    + (ovl <= p["sig_overlap"]), 0)
    s = pd.Series
    hh20 = s(H).shift(1).rolling(20, min_periods=20).max().to_numpy()
    ll20 = s(L).shift(1).rolling(20, min_periods=20).min().to_numpy()
    emasl = ema - np.r_[np.full(10, np.nan), ema[:-10]]
    above = (C > ema).astype(float)
    cross = np.r_[0, np.abs(np.diff(above))]
    side20 = ((np.abs(C - s(C).shift(20).to_numpy()) < atr) & (s(cross).rolling(20).sum().to_numpy() >= 3)
              & (s(ovl).rolling(20).mean().to_numpy() >= 0.5))
    ph2, pl2 = _pivots(np.ascontiguousarray(H), np.ascontiguousarray(L), 2)
    # trend strength TS (signed; positive = bull), components z-scored on a rolling 500-bar history
    rb, rs_ = _run_len(btb.astype(np.bool_)), _run_len(stb.astype(np.bool_))
    mg_b = np.r_[False, False, sb[1:-1] & (L[2:] > H[:-2])]
    mg_s = np.r_[False, False, ss[1:-1] & (H[2:] < L[:-2])]
    comp = [s(btb.astype(float) - stb).rolling(20).sum() / 20,
            s(above).rolling(20).mean() - 0.5,
            (s(rb).rolling(20).max() - s(rs_).rolling(20).max()) / 5,
            -(s(ovl).rolling(20).mean() - 0.5) * np.sign(C - s(C).shift(20)),
            s(np.clip((C - s(C).shift(20).to_numpy()) / atr, -5, 5)),
            s(mg_b.astype(float) - mg_s).rolling(20).sum()]
    z = [(c - c.rolling(500, min_periods=100).mean()) / c.rolling(500, min_periods=100).std() for c in comp]
    TS = pd.concat(z, axis=1).mean(axis=1).fillna(0.0).to_numpy()
    # sessions / days
    idx = pd.DatetimeIndex(df.index)
    if session == "rth":
        loc = idx.tz_convert("America/New_York")
        dkey = loc.normalize()
    else:
        dkey = idx.floor("D")
    day = pd.factorize(dkey)[0].astype(np.int64)
    first = np.r_[True, day[1:] != day[:-1]]
    last = np.r_[day[1:] != day[:-1], True]
    sbar = (np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0))) if session == "rth" else np.full(n, -1)
    g = pd.DataFrame({"d": day, "h": H, "l": L, "c": C, "o": O})
    dh, dl, dc = g.groupby("d").h.max(), g.groupby("d").l.min(), g.groupby("d").c.last()
    do_ = g.groupby("d").o.first()
    pdh = dh.shift(1).reindex(day).to_numpy()
    pdl = dl.shift(1).reindex(day).to_numpy()
    pcl = dc.shift(1).reindex(day).to_numpy()
    dop = do_.reindex(day).to_numpy()
    FE = np.zeros((n, NF))
    cols = {C_O: O, C_H: H, C_L: L, C_C: C, C_ATR: atr, C_ABR: abr, C_EMA: ema, C_R: R, C_DOJI: doji, C_BTB: btb,
            C_STB: stb, C_SB: sb, C_SS: ss, C_SURU: suru, C_SURD: surd, C_INS: ins, C_OVL: ovl, C_SIGB: sigb,
            C_SIGS: sigs, C_CPOS: cpos, C_HH20: hh20, C_LL20: ll20, C_EMASL: emasl, C_SIDE20: side20, C_PH2: ph2,
            C_PL2: pl2, C_TS: TS, C_DAY: day, C_SBAR: sbar, C_PDH: pdh, C_PDL: pdl, C_DOP: dop, C_PCL: pcl,
            C_SLAST: last if session == "rth" else np.zeros(n, bool),
            C_LEFT: (pd.Series(np.where(last, np.arange(n), np.nan)).bfill().to_numpy() - np.arange(n))
            if session == "rth" else np.full(n, -1.0)}
    for k, v in cols.items():
        FE[:, k] = np.asarray(v, float)
    return FE


# --------------------------------------------------------------------------- the engine
# OUT (n x 10 float): 0 state 1 ai 2 box_hi 3 box_lo 4 box_age 5 trend_age 6 nb_leg 7 ns_leg 8 barbwire 9 last pivot low
# 10 last pivot high (prices of the latest CONFIRMED k=2 pivots at bar t)
NOUT = 11
# events (claims): EV[kind] rows
EV_H, EV_BO, EV_FBO, EV_REV, EV_MTR, EV_FF, EV_MC, EV_CH, EV_SW = range(9)
NEV = 9
EVW = 10


@njit(cache=True)
def _add_ev(EV, cnt, kind, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    k = cnt[kind]
    if k < EV.shape[1]:
        EV[kind, k, 0] = a0; EV[kind, k, 1] = a1; EV[kind, k, 2] = a2; EV[kind, k, 3] = a3; EV[kind, k, 4] = a4
        EV[kind, k, 5] = a5; EV[kind, k, 6] = a6; EV[kind, k, 7] = a7; EV[kind, k, 8] = a8; EV[kind, k, 9] = a9
        cnt[kind] = k + 1


@njit(cache=True)
def _emit(IT, GATE, t, f, side, etype, epx, sta, stb, tga, tgb, fra, trb, tst, p0, te, be):
    IT[t, f, 0] = side; IT[t, f, 1] = etype; IT[t, f, 2] = epx; IT[t, f, 3] = sta; IT[t, f, 4] = stb
    IT[t, f, 5] = tga; IT[t, f, 6] = tgb; IT[t, f, 7] = fra; IT[t, f, 8] = trb; IT[t, f, 9] = tst
    IT[t, f, 10] = p0; IT[t, f, 11] = te; IT[t, f, 12] = be


@njit(cache=True)
def _room(side, epx, tgt, t, FE, box_hi, box_lo, in_tr, lph, lpl):
    """distance to the nearest opposing magnet >= distance to target (prior day H/L, day open, EMA, TR edge, pivot)"""
    d_t = abs(tgt - epx)
    best = 1e18
    cand = (FE[t, C_PDH], FE[t, C_PDL], FE[t, C_DOP], FE[t, C_EMA], box_hi if in_tr else np.nan,
            box_lo if in_tr else np.nan, lph, lpl)
    for m in cand:
        if m == m:
            dm = (m - epx) * side
            if dm > 1e-12 and dm < best:
                best = dm
    return best >= d_t


@njit(cache=True)
def engine(FE, P, tick, is_rth, OUT, IT, GATE, EV, evn):
    n = FE.shape[0]
    p_sig = P[K_SIG]
    p_zz, p_tlb, p_emat, p_dt, p_wt = P[K_ZZ], P[K_TLB], P[K_EMAT], P[K_DT], P[K_WT]
    wmin, wmax = int(P[K_WMIN]), int(P[K_WMAX])
    p_srun, p_smove, tr_stall, tr_min = int(P[K_SRUN]), P[K_SMOVE], int(P[K_STALL]), int(P[K_TRMIN])
    p_tight, p_broad, p_boc, bo_k = P[K_TIGHT], P[K_BROAD], P[K_BOC], int(P[K_BOK])
    p_mc, p_cmc, p_gap, p_htol, p_aic = int(P[K_MC]), int(P[K_CMC]), int(P[K_GAP]), P[K_HTOL], P[K_AIC]
    p_mtrt, mtr_exp = P[K_MTRT], int(P[K_MTRX])
    ff_bars, ff_h, ff_age, sec_k, use_room = int(P[K_FFB]), P[K_FFH], int(P[K_FFA]), int(P[K_SEC]), P[K_ROOM]
    ts_sw, ts_sc, sess_or = int(P[K_TSW]), int(P[K_TSC]), int(P[K_SOR])
    p0s = P[K_P0:K_P0 + 6]
    p_late, f4w, f4r, p_spb, p_orm = int(P[K_LATE]), int(P[K_F4W]), P[K_F4R], int(P[K_SPB]), P[K_ORM]
    # ---- state
    st = 0; ai = 0
    box_hi = np.nan; box_lo = np.nan; box_age = 0
    bo_side = 0; bo_t = -1; bo_lvl = 0.0; bo_ext = 0.0; bo_hi0 = 0.0; bo_lo0 = 0.0
    trend_start = -1; trend_ext = 0.0; since_ext = 0; legs = 0
    run_b = 0; run_b0 = 0.0; run_s = 0; run_s0 = 0.0
    spike_t = -1; mc_b_prev = 0; mc_s_prev = 0
    mc_b = 0; mc_s = 0
    gr_b = 0; gr_s = 0; last_gr_b = 0; last_gr_s = 0; gap_first_b = False; gap_first_s = False
    pend_ai_b = -1; pend_ai_s = -1
    # zig-zag
    zz_dir = 0; zz_ext = 0.0; zz_i = 0; sw_lo = np.nan; sw_hi = np.nan
    # pivots (ring buffers, most recent last)
    phx = np.full(6, np.nan); phi = np.full(6, -1.0); plx = np.full(6, np.nan); pli = np.full(6, -1.0)
    alt_px = np.full(6, np.nan); alt_ty = np.zeros(6)          # alternating pivot sequence (+1 high, -1 low)
    # counters (bull: highs in pullbacks; bear: lows)
    b_hi = -1e18; b_lo = 1e18; b_nl = 0; b_nb = 0; b_arm = False; b_val = False; b_sig = 1e18; b_lst = 1e18; b_pbl = 1e18
    s_lo = 1e18; s_hi = -1e18; s_nl = 0; s_nb = 0; s_arm = False; s_val = False; s_sig = -1e18; s_hst = -1e18; s_pbh = -1e18
    b_ext_t = 0; s_ext_t = 0; gr_b_end = -10**9; gr_s_end = -10**9; mt_top_rec = False; mt_bot_rec = False
    ff_rec = False
    # MTR / climax / final flag / F4 second entries / F3 / F6
    mt_top = -1; mt_hi = 0.0; mt_bot = -1; mt_lo = 0.0
    cl_up = -1; cl_dn = -1
    ff_side = 0; ff_t = -1; ff_hi = 0.0; ff_lo = 0.0
    f4_last_b = -10**9; f4_last_s = -10**9
    fb_side = 0; fb_t = -1; fb_ext = 0.0
    or_hi = -1e18; or_lo = 1e18; or_done = False; or_bo = 0; or_bt = -1
    ch_id_t = -1; ch_side = 0; ch_top = 0.0; ch_bot = 0.0
    f4b_t1 = -1; f4b_hi = 0.0; f4b_lo = 0.0; f4b_bn = False; f4b_rt = False      # F4 bull first entry
    f4s_t1 = -1; f4s_hi = 0.0; f4s_lo = 0.0; f4s_bn = False; f4s_rt = False      # F4 bear first entry
    sess_hi = -1e18; sess_lo = 1e18
    for t in range(2, n):
        O = FE[t, C_O]; H = FE[t, C_H]; L = FE[t, C_L]; C = FE[t, C_C]
        A = FE[t, C_ATR]; ABR = FE[t, C_ABR]; E = FE[t, C_EMA]
        Hp = FE[t - 1, C_H]; Lp = FE[t - 1, C_L]
        sbar_t = int(FE[t, C_SBAR])
        if is_rth and sbar_t == 0:
            st = 0; ai = 0; box_hi = np.nan; box_lo = np.nan; box_age = 0; bo_side = 0
            trend_start = -1; trend_ext = 0.0; since_ext = 0; legs = 0; spike_t = -1
            run_b = 0; run_s = 0; mc_b = 0; mc_s = 0; mc_b_prev = 0; mc_s_prev = 0
            gr_b = 0; gr_s = 0; last_gr_b = 0; last_gr_s = 0; gr_b_end = -10**9; gr_s_end = -10**9
            pend_ai_b = -1; pend_ai_s = -1
            zz_dir = 0; sw_lo = np.nan; sw_hi = np.nan
            phx[:] = np.nan; phi[:] = -1.0; plx[:] = np.nan; pli[:] = -1.0; alt_px[:] = np.nan; alt_ty[:] = 0.0
            b_val = False; b_arm = False; b_nl = 0; b_nb = 0; s_val = False; s_arm = False; s_nl = 0; s_nb = 0
            mt_top = -1; mt_bot = -1; cl_up = -1; cl_dn = -1; ff_side = 0; fb_side = 0
            f4b_t1 = -1; f4s_t1 = -1; f4_last_b = -10**9; f4_last_s = -10**9
            sess_hi = H; sess_lo = L
        elif is_rth:
            if H > sess_hi:
                sess_hi = H
            if L < sess_lo:
                sess_lo = L
        if not (A == A) or not (ABR == ABR) or A <= 0:
            continue
        # ---------------- pivots confirmed at t (k=2: pivot at t-2)
        j = t - 2
        if is_rth and FE[j, C_DAY] != FE[t, C_DAY]:
            j = -1
        if j >= 0 and FE[j, C_PH2] > 0:
            for q in range(5):
                phx[q] = phx[q + 1]; phi[q] = phi[q + 1]
            phx[5] = FE[j, C_H]; phi[5] = j
            if alt_ty[5] == 1 and FE[j, C_H] >= alt_px[5]:
                alt_px[5] = FE[j, C_H]
            elif alt_ty[5] != 1:
                for q in range(5):
                    alt_px[q] = alt_px[q + 1]; alt_ty[q] = alt_ty[q + 1]
                alt_px[5] = FE[j, C_H]; alt_ty[5] = 1
        if j >= 0 and FE[j, C_PL2] > 0:
            for q in range(5):
                plx[q] = plx[q + 1]; pli[q] = pli[q + 1]
            plx[5] = FE[j, C_L]; pli[5] = j
            if alt_ty[5] == -1 and FE[j, C_L] <= alt_px[5]:
                alt_px[5] = FE[j, C_L]
            elif alt_ty[5] != -1:
                for q in range(5):
                    alt_px[q] = alt_px[q + 1]; alt_ty[q] = alt_ty[q + 1]
                alt_px[5] = FE[j, C_L]; alt_ty[5] = -1
        lph = phx[5]; lpl = plx[5]
        # ---------------- zig-zag (ATR retrace) -> legs, counter leg starts
        new_up_leg = False
        new_dn_leg = False
        if zz_dir == 0:
            zz_dir = 1 if C >= O else -1; zz_ext = H if zz_dir == 1 else L; zz_i = t
        elif zz_dir == 1:
            if H > zz_ext:
                zz_ext = H; zz_i = t
            elif zz_ext - L >= p_zz * A:
                sw_hi = zz_ext
                _add_ev(EV, evn, EV_SW, zz_i, 1, zz_ext, t, 0, 0, 0, 0, 0, 0)
                if ai == 1 and st != 0:
                    legs += 1
                zz_dir = -1; zz_ext = L; zz_i = t
                new_dn_leg = True
        else:
            if L < zz_ext:
                zz_ext = L; zz_i = t
            elif H - zz_ext >= p_zz * A:
                sw_lo = zz_ext
                _add_ev(EV, evn, EV_SW, zz_i, -1, zz_ext, t, 0, 0, 0, 0, 0, 0)
                if ai == -1 and st != 0:
                    legs += 1
                zz_dir = 1; zz_ext = H; zz_i = t
                new_up_leg = True
        # ---------------- counters: process yesterday's armed order (trigger at t)
        sig_w = FE[t - 1, C_SIGB]
        if b_arm and H >= Hp + tick:
            b_nb += 1
            cnt_ok = (b_nl == 0) or (b_lst <= b_sig + p_htol * A)
            if cnt_ok:
                b_nl += 1
            emat = 1.0 if b_pbl <= FE[t - 1, C_EMA] + p_emat * FE[t - 1, C_ATR] else 0.0
            _add_ev(EV, evn, EV_H, t - 1, 1, b_nl, b_nb, Hp + tick, Lp - tick, 1.0 if ai == 1 else 0.0, st, emat,
                    sig_w + 10.0 * min(t - 1 - b_ext_t, 999))
            b_sig = Lp; b_lst = 1e18
        sig_w = FE[t - 1, C_SIGS]
        if s_arm and L <= Lp - tick:
            s_nb += 1
            cnt_ok = (s_nl == 0) or (s_hst >= s_sig - p_htol * A)
            if cnt_ok:
                s_nl += 1
            emat = 1.0 if s_pbh >= FE[t - 1, C_EMA] - p_emat * FE[t - 1, C_ATR] else 0.0
            _add_ev(EV, evn, EV_H, t - 1, -1, s_nl, s_nb, Lp - tick, Hp + tick, 1.0 if ai == -1 else 0.0, st, emat,
                    sig_w + 10.0 * min(t - 1 - s_ext_t, 999))
            s_sig = Hp; s_hst = -1e18
        # leg starts from the zig-zag
        # a new zig-zag leg starts a count only if there is no valid one (a higher-low pullback keeps counting)
        if new_up_leg and not b_val:
            b_lo = sw_lo; b_hi = zz_ext; b_nl = 0; b_nb = 0; b_arm = False; b_val = True; b_lst = 1e18; b_pbl = 1e18
            b_ext_t = t
        if new_dn_leg and not s_val:
            s_hi = sw_hi; s_lo = zz_ext; s_nl = 0; s_nb = 0; s_arm = False; s_val = True; s_hst = -1e18; s_pbh = -1e18
            s_ext_t = t
        # bull counter update
        if b_val:
            if H > b_hi:
                if b_pbl < 1e17:
                    b_lo = b_pbl                                   # invalidation moves up to the last higher low
                b_hi = H; b_nl = 0; b_nb = 0; b_arm = False; b_lst = 1e18; b_pbl = 1e18; b_ext_t = t
            elif L < b_lo:
                b_val = False; b_arm = False; b_nl = 0; b_nb = 0
            else:
                b_arm = H <= Hp
                if L < b_lst:
                    b_lst = L
                if L < b_pbl:
                    b_pbl = L
            if b_nl > 4:
                b_nl = 0; b_nb = 0
        if s_val:
            if L < s_lo:
                if s_pbh > -1e17:
                    s_hi = s_pbh
                s_lo = L; s_nl = 0; s_nb = 0; s_arm = False; s_hst = -1e18; s_pbh = -1e18; s_ext_t = t
            elif H > s_hi:
                s_val = False; s_arm = False; s_nl = 0; s_nb = 0
            else:
                s_arm = L >= Lp
                if H > s_hst:
                    s_hst = H
                if H > s_pbh:
                    s_pbh = H
            if s_nl > 4:
                s_nl = 0; s_nb = 0
        b_next = (b_nl + 1) if (b_nl == 0 or b_lst <= b_sig + p_htol * A) else b_nl
        s_next = (s_nl + 1) if (s_nl == 0 or s_hst >= s_sig - p_htol * A) else s_nl
        # ---------------- runs, micro channels, MA gap bars
        if FE[t, C_SB] > 0:
            run_b += 1
            if run_b == 1:
                run_b0 = O
        else:
            run_b = 0
        if FE[t, C_SS] > 0:
            run_s += 1
            if run_s == 1:
                run_s0 = O
        else:
            run_s = 0
        mc_b_prev = mc_b; mc_s_prev = mc_s
        mc_b = mc_b + 1 if L >= Lp else 0
        mc_s = mc_s + 1 if H <= Hp else 0
        if L > E:
            gr_b += 1
        else:
            if gr_b > 0:
                last_gr_b = gr_b; gr_b_end = t
            gr_b = 0
        if H < E:
            gr_s += 1
        else:
            if gr_s > 0:
                last_gr_s = gr_s; gr_s_end = t
            gr_s = 0
        gap_first_b = gr_s == 1 and last_gr_b >= p_gap and t - gr_b_end <= 20     # first bar fully below the EMA
        gap_first_s = gr_b == 1 and last_gr_s >= p_gap and t - gr_s_end <= 20
        # ---------------- market cycle and always-in
        hh20 = FE[t, C_HH20]; ll20 = FE[t, C_LL20]
        ema_b = C > E
        ema_s = C < E
        if is_rth:
            if sbar_t >= 1:
                k0 = max(t - 20, t - sbar_t)
                rh = -1e18; rl = 1e18
                for q in range(k0, t):
                    if FE[q, C_H] > rh:
                        rh = FE[q, C_H]
                    if FE[q, C_L] < rl:
                        rl = FE[q, C_L]
                hh20 = rh; ll20 = rl
                if sbar_t < 20:                   # opening spikes count even on the "wrong" side of a lagging EMA
                    ema_b = True; ema_s = True
            else:
                hh20 = np.nan; ll20 = np.nan
        spike_up = hh20 == hh20 and C > hh20 and ((run_b >= p_srun and C - run_b0 >= p_smove * A) or FE[t, C_SURU] > 0)
        spike_dn = ll20 == ll20 and C < ll20 and ((run_s >= p_srun and run_s0 - C >= p_smove * A) or FE[t, C_SURD] > 0)
        if spike_up and ema_b:
            if st <= 0:
                trend_start = t; legs = 0; trend_ext = H; since_ext = 0
            st = 1; ai = 1; bo_side = 0; spike_t = t
        elif spike_dn and ema_s:
            if st >= 0:
                trend_start = t; legs = 0; trend_ext = L; since_ext = 0
            st = -1; ai = -1; bo_side = 0; spike_t = t
        else:
            # always-in flip by a strong breakout of the latest swing, confirmed by a follow-through bar
            if pend_ai_b == t - 1 and C > O and L >= (FE[t - 1, C_H] + FE[t - 1, C_L]) / 2:
                ai = 1
            if pend_ai_s == t - 1 and C < O and H <= (FE[t - 1, C_H] + FE[t - 1, C_L]) / 2:
                ai = -1
        if FE[t, C_SB] > 0 and lph == lph and C > lph:
            pend_ai_b = t
        if FE[t, C_SS] > 0 and lpl == lpl and C < lpl:
            pend_ai_s = t
        if st == 1 and t > spike_t and (L < Lp or FE[t, C_STB] > 0):
            st = 2; ch_id_t = t; ch_side = 1; ch_top = trend_ext; ch_bot = L
            _add_ev(EV, evn, EV_CH, t, 1, trend_ext, L, 0, 0, 0, 0, 0, 0)
        if st == -1 and t > spike_t and (H > Hp or FE[t, C_BTB] > 0):
            st = -2; ch_id_t = t; ch_side = -1; ch_top = H; ch_bot = trend_ext
            _add_ev(EV, evn, EV_CH, t, -1, H, trend_ext, 0, 0, 0, 0, 0, 0)
        if st > 0:
            if H > trend_ext:
                trend_ext = H; since_ext = 0
            else:
                since_ext += 1
        elif st < 0:
            if L < trend_ext:
                trend_ext = L; since_ext = 0
            else:
                since_ext += 1
        to_tr = False
        if (st == 2 or st == -2) and since_ext >= tr_stall and abs(FE[t, C_EMASL]) < 0.1 * A:
            to_tr = True
        if st != 0 and FE[t, C_SIDE20] > 0:
            to_tr = True
        if to_tr:
            st = 0; box_age = 20; bo_side = 0
            box_hi = -1e18; box_lo = 1e18
            for q in range(max(0, t - 19), t + 1):
                if FE[q, C_H] > box_hi:
                    box_hi = FE[q, C_H]
                if FE[q, C_L] < box_lo:
                    box_lo = FE[q, C_L]
        elif st == 0 and box_hi == box_hi:
            box_age += 1
            if bo_side == 0:
                if C > box_hi + p_boc * A:
                    bo_side = 1; bo_t = t; bo_lvl = box_hi; bo_ext = H; bo_hi0 = box_hi; bo_lo0 = box_lo
                    _add_ev(EV, evn, EV_BO, t, 1, box_hi, box_lo, 1.0 if box_hi - box_lo >= p_broad * A else 0.0,
                            1.0 if box_hi - box_lo <= p_tight * A else 0.0, box_age, A, 0, 0)
                elif C < box_lo - p_boc * A:
                    bo_side = -1; bo_t = t; bo_lvl = box_lo; bo_ext = L; bo_hi0 = box_hi; bo_lo0 = box_lo
                    _add_ev(EV, evn, EV_BO, t, -1, box_hi, box_lo, 1.0 if box_hi - box_lo >= p_broad * A else 0.0,
                            1.0 if box_hi - box_lo <= p_tight * A else 0.0, box_age, A, 0, 0)
                else:
                    if H > box_hi:
                        box_hi = H
                    if L < box_lo:
                        box_lo = L
            else:
                k = t - bo_t
                if bo_side == 1:
                    if H > bo_ext:
                        bo_ext = H
                    if k == 1 and C > O and L >= (FE[bo_t, C_H] + FE[bo_t, C_L]) / 2:
                        st = 1; ai = 1; trend_start = bo_t; trend_ext = bo_ext; since_ext = 0; legs = 0; bo_side = 0
                    elif C < bo_lvl and k <= bo_k:
                        _add_ev(EV, evn, EV_FBO, t, 1, bo_t, bo_ext, bo_hi0, bo_lo0, 0, 0, 0, 0)
                        fb_side = 1; fb_t = t; fb_ext = bo_ext; bo_side = 0
                    elif k >= bo_k:
                        st = 2; ai = 1; trend_start = bo_t; trend_ext = bo_ext; since_ext = 0; legs = 0; bo_side = 0
                else:
                    if L < bo_ext:
                        bo_ext = L
                    if k == 1 and C < O and H <= (FE[bo_t, C_H] + FE[bo_t, C_L]) / 2:
                        st = -1; ai = -1; trend_start = bo_t; trend_ext = bo_ext; since_ext = 0; legs = 0; bo_side = 0
                    elif C > bo_lvl and k <= bo_k:
                        _add_ev(EV, evn, EV_FBO, t, -1, bo_t, bo_ext, bo_hi0, bo_lo0, 0, 0, 0, 0)
                        fb_side = -1; fb_t = t; fb_ext = bo_ext; bo_side = 0
                    elif k >= bo_k:
                        st = -2; ai = -1; trend_start = bo_t; trend_ext = bo_ext; since_ext = 0; legs = 0; bo_side = 0
        in_tr = st == 0 and box_hi == box_hi and box_age >= tr_min
        bw = (FE[t, C_OVL] >= 0.67 and FE[t - 1, C_OVL] >= 0.67 and
              (FE[t, C_DOJI] + FE[t - 1, C_DOJI] + FE[t - 2, C_DOJI]) >= 1 and
              (FE[t, C_R] + FE[t - 1, C_R] + FE[t - 2, C_R]) / 3 >= 0.8 * ABR)
        trend_age = t - trend_start if trend_start >= 0 else 0
        # ---------------- micro-channel and reversal-attempt events (claims)
        if mc_b == p_cmc:
            _add_ev(EV, evn, EV_MC, t, 1, mc_b, H, A, 0, 0, 0, 0, 0)
        if mc_s == p_cmc:
            _add_ev(EV, evn, EV_MC, t, -1, mc_s, L, A, 0, 0, 0, 0, 0)
        if mc_b == p_mc:
            _add_ev(EV, evn, EV_MC, t, 1, mc_b, H, A, 1, 0, 0, 0, 0)
        if mc_s == p_mc:
            _add_ev(EV, evn, EV_MC, t, -1, mc_s, L, A, 1, 0, 0, 0, 0)
        if ai == 1 and FE[t, C_TS] > 0 and FE[t, C_SIGS] >= p_sig:
            _add_ev(EV, evn, EV_REV, t, -1, L - tick, trend_ext, A, 0, 0, 0, 0, 0)
        if ai == -1 and FE[t, C_TS] < 0 and FE[t, C_SIGB] >= p_sig:
            _add_ev(EV, evn, EV_REV, t, 1, H + tick, trend_ext, A, 0, 0, 0, 0, 0)
        # ---------------- climaxes
        if ai == 1 and ((FE[t, C_SURU] > 0 and trend_age >= 20) or mc_b >= p_cmc):
            cl_up = t
        if ai == -1 and ((FE[t, C_SURD] > 0 and trend_age >= 20) or mc_s >= p_cmc):
            cl_dn = t
        if phi[4] >= 0 and phi[5] > phi[4] and phx[5] > phx[4]:          # trend channel line overshoot (bull)
            slope = (phx[5] - phx[4]) / (phi[5] - phi[4])
            if Hp > phx[5] + slope * (t - 1 - phi[5]) and C < phx[5] + slope * (t - phi[5]):
                cl_up = t
        if pli[4] >= 0 and pli[5] > pli[4] and plx[5] < plx[4]:
            slope = (plx[5] - plx[4]) / (pli[5] - pli[4])
            if Lp < plx[5] + slope * (t - 1 - pli[5]) and C > plx[5] + slope * (t - pli[5]):
                cl_dn = t
        climax_up = cl_up >= 0 and t - cl_up <= 20
        climax_dn = cl_dn >= 0 and t - cl_dn <= 20
        # ---------------- MTR arming (trend-line break or long pullback to the EMA, then a test of the extreme)
        if ai == 1 and st != 0 and legs >= 2 and pli[4] >= trend_start and plx[5] > plx[4] and mt_top < 0:
            line = plx[4] + (plx[5] - plx[4]) / (pli[5] - pli[4]) * (t - pli[4])
            if C < line - p_tlb * A or (since_ext >= 5 and L <= E):
                mt_top = t; mt_hi = trend_ext; mt_top_rec = False
        if ai == -1 and st != 0 and legs >= 2 and phi[4] >= trend_start and phx[5] < phx[4] and mt_bot < 0:
            line = phx[4] + (phx[5] - phx[4]) / (phi[5] - phi[4]) * (t - phi[4])
            if C > line + p_tlb * A or (since_ext >= 5 and H >= E):
                mt_bot = t; mt_lo = trend_ext; mt_bot_rec = False
        if mt_top >= 0 and (t - mt_top > mtr_exp or H > mt_hi + A):
            mt_top = -1
        if mt_bot >= 0 and (t - mt_bot > mtr_exp or L < mt_lo - A):
            mt_bot = -1
        # ---------------- final flag tracking (tight box of the last ff_bars bars late in a trend)
        bh = -1e18; bl = 1e18
        for q in range(t - ff_bars, t):
            if FE[q, C_H] > bh:
                bh = FE[q, C_H]
            if FE[q, C_L] < bl:
                bl = FE[q, C_L]
        late = trend_age >= ff_age or legs >= 2
        if ff_side == 0 and late and bh - bl <= ff_h * A:
            if ai == 1 and C > bh + p_boc * A:
                ff_side = 1; ff_t = t; ff_hi = bh; ff_lo = bl; ff_rec = False
            elif ai == -1 and C < bl - p_boc * A:
                ff_side = -1; ff_t = t; ff_hi = bh; ff_lo = bl; ff_rec = False
        if ff_side != 0 and t - ff_t > bo_k:
            ff_side = 0
        # ================= SETUPS (signal bar = t; orders act on bar t+1) =================
        sigB = FE[t, C_SIGB]; sigS = FE[t, C_SIGS]; cpos = FE[t, C_CPOS]
        good_b = sigB >= p_sig; good_s = sigS >= p_sig
        TS = FE[t, C_TS]
        n_bt = 0; n_st = 0; n_sb = 0; n_ss = 0
        for q in range(t - 4, t + 1):
            n_bt += 1 if FE[q, C_BTB] > 0 else 0
            n_st += 1 if FE[q, C_STB] > 0 else 0
            n_sb += 1 if FE[q, C_SB] > 0 else 0
            n_ss += 1 if FE[q, C_SS] > 0 else 0
        spiky_up = n_sb >= p_spb or n_bt >= 4
        spiky_dn = n_ss >= p_spb or n_st >= 4
        # pattern detectors
        db = False; dt = False
        if plx[5] == plx[5] and t - pli[5] <= 80 and t - pli[5] >= 3 and phi[5] > pli[5]:
            if abs(min(L, Lp) - plx[5]) <= p_dt * A:
                db = True
        if phx[5] == phx[5] and t - phi[5] <= 80 and t - phi[5] >= 3 and pli[5] > phi[5]:
            if abs(max(H, Hp) - phx[5]) <= p_dt * A:
                dt = True
        mdb = False; mdt = False
        for q in range(1, 4):
            if abs(L - FE[t - q, C_L]) <= 0.1 * A:
                mdb = True
            if abs(H - FE[t - q, C_H]) <= 0.1 * A:
                mdt = True
        wb = False; wt = False
        if pli[4] >= 0 and plx[5] < plx[4] and L < plx[5] + p_wt * A and wmin <= t - pli[4] <= wmax and phi[5] > pli[4]:
            wb = True
        if phi[4] >= 0 and phx[5] > phx[4] and H > phx[5] - p_wt * A and wmin <= t - phi[4] <= wmax and pli[5] > phi[4]:
            wt = True
        etb = False; ett = False
        if alt_ty[1] != 0:
            hs = np.full(3, np.nan); ls_ = np.full(3, np.nan); kh = 0; kl = 0
            for q in range(1, 6):
                if alt_ty[q] == 1 and kh < 3:
                    hs[kh] = alt_px[q]; kh += 1
                elif alt_ty[q] == -1 and kl < 3:
                    ls_[kl] = alt_px[q]; kl += 1
            if kh >= 2 and kl >= 2 and hs[kh - 1] > hs[kh - 2] and ls_[kl - 1] < ls_[kl - 2]:
                if L < ls_[kl - 1] and good_b:
                    etb = True
                if H > hs[kh - 1] and good_s:
                    ett = True
        box_mid = (box_hi + box_lo) / 2 if box_hi == box_hi else np.nan
        third = (box_hi - box_lo) / 3 if box_hi == box_hi else 0.0
        # ---------------- context gates (for the context-only benchmark B1)
        GATE[t, 0] = 1 if (ai == 1 and TS >= p_aic and (st == 1 or st == 2)) else (-1 if (ai == -1 and TS <= -p_aic and (st == -1 or st == -2)) else 0)
        if in_tr and not bw:
            GATE[t, 1] = 1 if C <= box_lo + third else (-1 if C >= box_hi - third else 0)
        if fb_side != 0 and t - fb_t <= bo_k:
            GATE[t, 2] = -fb_side
        GATE[t, 3] = -1 if (mt_top >= 0 or climax_up) else (1 if (mt_bot >= 0 or climax_dn) else 0)
        GATE[t, 4] = 1 if (st == 1) else (-1 if st == -1 else 0)
        # ---------------- F1 with-trend H2/L2 (H1/L1 in a spike or micro channel)
        if not bw:
            if GATE[t, 0] == 1 and b_val and b_arm and good_b and (FE[t, C_BTB] > 0 or cpos >= 0.6):
                emat = b_pbl <= E + p_emat * A
                ok = (b_next == 2 and (emat or gap_first_b)) or (b_next == 1 and (st == 1 or mc_b_prev >= p_mc)) or gap_first_b
                if ok:
                    e = H + tick; sa = L - tick; sb_ = min(b_pbl, L) - tick
                    mm = b_pbl + (b_hi - b_lo)
                    tgb = max(b_hi, mm) if max(b_hi, mm) > e else e + 2 * (e - sb_)
                    te = 1.0 if (p0s[0] * 1.0 - (1 - p0s[0]) > 0) and (use_room < 0.5 or _room(1, e, e + (e - sa), t, FE, box_hi, box_lo, in_tr, lph, lpl)) else 0.0
                    _emit(IT, GATE, t, 0, 1, T_STOP, e, sa, sb_, e + (e - sa), tgb, 0.5, 1, ts_sw, p0s[0], te, 1)
            if GATE[t, 0] == -1 and s_val and s_arm and good_s and (FE[t, C_STB] > 0 or cpos <= 0.4):
                emat = s_pbh >= E - p_emat * A
                ok = (s_next == 2 and (emat or gap_first_s)) or (s_next == 1 and (st == -1 or mc_s_prev >= p_mc)) or gap_first_s
                if ok:
                    e = L - tick; sa = H + tick; sb_ = max(s_pbh, H) + tick
                    mm = s_pbh - (s_hi - s_lo)
                    tgb = min(s_lo, mm) if min(s_lo, mm) < e else e - 2 * (sb_ - e)
                    te = 1.0 if (use_room < 0.5 or _room(-1, e, e - (sa - e), t, FE, box_hi, box_lo, in_tr, lph, lpl)) else 0.0
                    _emit(IT, GATE, t, 0, -1, T_STOP, e, sa, sb_, e - (sa - e), tgb, 0.5, 1, ts_sw, p0s[0], te, 1)
        # ---------------- F2 trading-range fade (buy low, sell high, scalp)
        if GATE[t, 1] == 1 and good_b and not spiky_dn and (b_next == 2 or db or wb or mdb):
            e = H + tick; sa = L - tick; tg = min(e + (e - sa), box_mid)
            if tg > e:
                rr = (tg - e) / (e - sa)
                te = 1.0 if p0s[1] * rr - (1 - p0s[1]) > 0 and (use_room < 0.5 or _room(1, e, tg, t, FE, np.nan, np.nan, False, lph, lpl)) else 0.0
                _emit(IT, GATE, t, 1, 1, T_STOP, e, sa, sa, tg, tg, 1.0, 0, ts_sc, p0s[1], te, 1)
        if GATE[t, 1] == -1 and good_s and not spiky_up and (s_next == 2 or dt or wt or mdt):
            e = L - tick; sa = H + tick; tg = max(e - (sa - e), box_mid)
            if tg < e:
                rr = (e - tg) / (sa - e)
                te = 1.0 if p0s[1] * rr - (1 - p0s[1]) > 0 and (use_room < 0.5 or _room(-1, e, tg, t, FE, np.nan, np.nan, False, lph, lpl)) else 0.0
                _emit(IT, GATE, t, 1, -1, T_STOP, e, sa, sa, tg, tg, 1.0, 0, ts_sc, p0s[1], te, 1)
        # ---------------- F3 failed breakout of a broad range (reversal bar closes back inside)
        if fb_side != 0 and t - fb_t <= bo_k and bo_hi0 - bo_lo0 >= p_broad * A:
            mid = (bo_hi0 + bo_lo0) / 2
            if fb_side == -1 and good_b and C > bo_lo0:
                e = H + tick; sa = fb_ext - tick
                if e - sa > 0:
                    rr = (mid - e) / (e - sa)
                    te = 1.0 if (mid > e and p0s[2] * rr - (1 - p0s[2]) > 0 and
                                 (use_room < 0.5 or _room(1, e, mid, t, FE, np.nan, np.nan, False, np.nan, np.nan))) else 0.0
                    _emit(IT, GATE, t, 2, 1, T_STOP, e, sa, sa, mid, bo_hi0, 0.5, 0, ts_sw, p0s[2], te, 1)
                    fb_side = 0
            elif fb_side == 1 and good_s and C < bo_hi0:
                e = L - tick; sa = fb_ext + tick
                if sa - e > 0:
                    rr = (e - mid) / (sa - e)
                    te = 1.0 if (mid < e and p0s[2] * rr - (1 - p0s[2]) > 0 and
                                 (use_room < 0.5 or _room(-1, e, mid, t, FE, np.nan, np.nan, False, np.nan, np.nan))) else 0.0
                    _emit(IT, GATE, t, 2, -1, T_STOP, e, sa, sa, mid, bo_lo0, 0.5, 0, ts_sw, p0s[2], te, 1)
                    fb_side = 0
        # ---------------- F4 reversals: MTR, wedge, DT/DB at range extremes, expanding triangle, final flag
        bull_setup = good_b and (
            (mt_bot >= 0 and L <= mt_lo + p_mtrt * A) or
            (wb and (climax_dn or (in_tr and C <= box_lo + third) or mt_bot >= 0)) or
            (db and in_tr and C <= box_lo + third) or etb or
            (ff_side == -1 and t - ff_t <= bo_k and C > ff_lo))
        bear_setup = good_s and (
            (mt_top >= 0 and H >= mt_hi - p_mtrt * A) or
            (wt and (climax_up or (in_tr and C >= box_hi - third) or mt_top >= 0)) or
            (dt and in_tr and C >= box_hi - third) or ett or
            (ff_side == 1 and t - ff_t <= bo_k and C < ff_hi))
        # the re-test may be the new signal bar itself; the bounce (first entry triggered) must come before it
        if f4b_t1 >= 0:
            if t - f4b_t1 > f4w:
                f4b_t1 = -1
            elif t > f4b_t1 and f4b_bn and L <= f4b_lo + f4r * A:
                f4b_rt = True
        if f4s_t1 >= 0:
            if t - f4s_t1 > f4w:
                f4s_t1 = -1
            elif t > f4s_t1 and f4s_bn and H >= f4s_hi - f4r * A:
                f4s_rt = True
        if bull_setup:
            if mt_bot >= 0 and L <= mt_lo + p_mtrt * A and not mt_bot_rec:
                _add_ev(EV, evn, EV_MTR, t, 1, H + tick, min(L, Lp) - tick, 0, 0, 0, 0, 0, 0)
                mt_bot_rec = True
            if ff_side == -1 and not ff_rec:
                _add_ev(EV, evn, EV_FF, t, 1, H + tick, min(L, Lp) - tick, 0, 0, 0, 0, 0, 0)
                ff_rec = True
            if f4b_t1 >= 0 and f4b_bn and f4b_rt and t - f4b_t1 >= 2:
                e = H + tick; sa = min(L, Lp, FE[t - 2, C_L]) - tick; tg = e + 2 * (e - sa)
                te = 1.0 if (p0s[3] * 2 - (1 - p0s[3]) > 0) and (use_room < 0.5 or _room(1, e, tg, t, FE, box_hi, box_lo, in_tr, lph, np.nan)) else 0.0
                _emit(IT, GATE, t, 3, 1, T_STOP, e, sa, sa, tg, tg, 1.0, 1, ts_sw, p0s[3], te, 1)
                f4b_t1 = -1
                if mt_bot >= 0:
                    mt_bot = -1
            elif f4b_t1 < 0 or not f4b_bn:                     # (re)start the first entry; adjacent bars are not 2nd
                f4b_t1 = t; f4b_hi = H; f4b_lo = min(L, Lp); f4b_bn = False; f4b_rt = False
        if bear_setup:
            if mt_top >= 0 and H >= mt_hi - p_mtrt * A and not mt_top_rec:
                _add_ev(EV, evn, EV_MTR, t, -1, L - tick, max(H, Hp) + tick, 0, 0, 0, 0, 0, 0)
                mt_top_rec = True
            if ff_side == 1 and not ff_rec:
                _add_ev(EV, evn, EV_FF, t, -1, L - tick, max(H, Hp) + tick, 0, 0, 0, 0, 0, 0)
                ff_rec = True
            if f4s_t1 >= 0 and f4s_bn and f4s_rt and t - f4s_t1 >= 2:
                e = L - tick; sa = max(H, Hp, FE[t - 2, C_H]) + tick; tg = e - 2 * (sa - e)
                te = 1.0 if (use_room < 0.5 or _room(-1, e, tg, t, FE, box_hi, box_lo, in_tr, np.nan, lpl)) else 0.0
                _emit(IT, GATE, t, 3, -1, T_STOP, e, sa, sa, tg, tg, 1.0, 1, ts_sw, p0s[3], te, 1)
                f4s_t1 = -1
                if mt_top >= 0:
                    mt_top = -1
            elif f4s_t1 < 0 or not f4s_bn:
                f4s_t1 = t; f4s_hi = max(H, Hp); f4s_lo = L; f4s_bn = False; f4s_rt = False
        if f4b_t1 >= 0 and t > f4b_t1 and H >= f4b_hi + tick:
            f4b_bn = True
        if f4s_t1 >= 0 and t > f4s_t1 and L <= f4s_lo - tick:
            f4s_bn = True
        # ---------------- F5 strong breakout: buy the close of the follow-through bar (bar t follows breakout bar b)
        b = t - 1
        if b >= 7 and OUT[b - 1, 0] == 0 and OUT[b - 1, 4] >= 5:
            lvl = OUT[b - 1, 2]; lvl2 = OUT[b - 1, 3]
            Hb = FE[b, C_H]; Lb = FE[b, C_L]; Cb = FE[b, C_C]; Rb = FE[b, C_R]; Ab = FE[b, C_ATR]
            for side in (1, -1):
                brk = (Cb > lvl + p_boc * Ab) if side == 1 else (Cb < lvl2 - p_boc * Ab)
                if not brk:
                    continue
                sc = 0
                if side == 1:
                    sc += 1 if (FE[b, C_SB] > 0 and Rb >= 1.5 * FE[b, C_ABR]) else 0
                    sc += 1 if FE[b, C_CPOS] >= 0.75 else 0
                    sc += 1 if L > lvl else 0                                   # measuring gap
                    sc += 1 if (C > O and L >= (Hb + Lb) / 2) else 0            # follow-through
                    mx = -1e18
                    for q in range(b - 5, b):
                        mx = max(mx, FE[q, C_H], FE[q, C_C])
                    sc += 1 if Cb > mx else 0
                else:
                    sc += 1 if (FE[b, C_SS] > 0 and Rb >= 1.5 * FE[b, C_ABR]) else 0
                    sc += 1 if FE[b, C_CPOS] <= 0.25 else 0
                    sc += 1 if H < lvl2 else 0
                    sc += 1 if (C < O and H <= (Hb + Lb) / 2) else 0
                    mn = 1e18
                    for q in range(b - 5, b):
                        mn = min(mn, FE[q, C_L], FE[q, C_C])
                    sc += 1 if Cb < mn else 0
                if sc >= 4:
                    e = C
                    sa = (Lb - tick) if side == 1 else (Hb + tick)
                    tg = (lvl + (lvl - lvl2)) if side == 1 else (lvl2 - (lvl - lvl2))
                    risk = (e - sa) * side
                    gain = (tg - e) * side
                    if risk > 0 and gain > 0:
                        rr = gain / risk
                        p0 = p0s[4] if rr <= 1 else 0.4
                        te = 1.0 if p0 * rr - (1 - p0) > 0 and (use_room < 0.5 or _room(side, e, tg, t, FE, np.nan, np.nan, False,
                                                                                          lph if side == -1 else np.nan,
                                                                                          lpl if side == 1 else np.nan)) else 0.0
                        _emit(IT, GATE, t, 4, side, T_MARKET, e, sa, sa, tg, tg, 1.0, 0, ts_sw, p0, te, 1)
        # ---------------- F6 opening patterns (US session only)
        if is_rth:
            sb_ = int(FE[t, C_SBAR])
            if sb_ == 0:
                or_hi = H; or_lo = L; or_done = False; or_bo = 0; or_bt = -1
            elif sb_ < sess_or:
                or_hi = max(or_hi, H); or_lo = min(or_lo, L)
            left = int(FE[t, C_LEFT])
            if 0 <= sb_ < sess_or and left >= 2:
                mags = (FE[t, C_PDL], FE[t, C_PCL], E, FE[t, C_PDH])
                dop = FE[t, C_DOP]
                moved_dn = dop - sess_lo >= p_orm * A and L <= sess_lo + 0.1 * A
                moved_up = sess_hi - dop >= p_orm * A and H >= sess_hi - 0.1 * A
                near_b = False; near_s = False
                for m in mags:
                    if m == m:
                        if moved_dn and abs(L - m) <= 0.5 * A:
                            near_b = True
                        if moved_up and abs(H - m) <= 0.5 * A:
                            near_s = True
                GATE[t, 5] = 1 if near_b else (-1 if near_s else 0)
                if near_b and good_b and sb_ >= 1:
                    e = H + tick; sa = L - tick
                    _emit(IT, GATE, t, 5, 1, T_STOP, e, sa, sa, e + 2 * (e - sa), e + 2 * (e - sa), 1.0, 0, left, p0s[5],
                          1.0 if p0s[5] * 2 - (1 - p0s[5]) > 0 else 0.0, 1)
                elif near_s and good_s and sb_ >= 1:
                    e = L - tick; sa = H + tick
                    _emit(IT, GATE, t, 5, -1, T_STOP, e, sa, sa, e - 2 * (sa - e), e - 2 * (sa - e), 1.0, 0, left, p0s[5],
                          1.0 if p0s[5] * 2 - (1 - p0s[5]) > 0 else 0.0, 1)
            if sb_ >= sess_or and left >= 2 and not or_done:
                if or_bo == 0:
                    if C > or_hi + p_boc * A:
                        or_bo = 1; or_bt = t
                    elif C < or_lo - p_boc * A:
                        or_bo = -1; or_bt = t
                elif t == or_bt + 1:
                    Hb = FE[or_bt, C_H]; Lb = FE[or_bt, C_L]
                    sc = (1 if FE[or_bt, C_SB if or_bo == 1 else C_SS] > 0 else 0) + \
                         (1 if (FE[or_bt, C_CPOS] >= 0.75 if or_bo == 1 else FE[or_bt, C_CPOS] <= 0.25) else 0) + \
                         (1 if (L > or_hi if or_bo == 1 else H < or_lo) else 0) + \
                         (1 if ((C > O and L >= (Hb + Lb) / 2) if or_bo == 1 else (C < O and H <= (Hb + Lb) / 2)) else 0) + \
                         (1 if FE[or_bt, C_R] >= 1.5 * FE[or_bt, C_ABR] else 0)
                    if sc >= 4:
                        e = C
                        sa = (Lb - tick) if or_bo == 1 else (Hb + tick)
                        if (e - sa) * or_bo > 0:
                            _emit(IT, GATE, t, 5, or_bo, T_MARKET, e, sa, sa, e + 2 * (e - sa), e + 2 * (e - sa), 1.0, 0, left,
                                  p0s[5], 1.0, 1)
                    or_done = True
        if is_rth and (sbar_t == 0 or FE[t, C_LEFT] < p_late):
            for f in range(6):
                for q in range(NI):
                    IT[t, f, q] = 0.0
                GATE[t, f] = 0.0
        # ---------------- outputs
        OUT[t, 0] = st; OUT[t, 1] = ai; OUT[t, 2] = box_hi; OUT[t, 3] = box_lo; OUT[t, 4] = box_age
        OUT[t, 5] = trend_age; OUT[t, 6] = b_nl; OUT[t, 7] = s_nl; OUT[t, 8] = 1.0 if bw else 0.0
        OUT[t, 9] = lpl; OUT[t, 10] = lph


def run_engine(FE, P, tick, is_rth, ev_cap=400_000):
    n = FE.shape[0]
    OUT = np.full((n, NOUT), np.nan)
    IT = np.zeros((n, 6, NI))
    GATE = np.zeros((n, 6))
    EV = np.zeros((NEV, ev_cap, EVW))
    evn = np.zeros(NEV, np.int64)
    engine(FE, P, float(tick), bool(is_rth), OUT, IT, GATE, EV, evn)
    events = {k: EV[i, :evn[i]].copy() for i, k in enumerate(["H", "BO", "FBO", "REV", "MTR", "FF", "MC", "CH", "SW"])}
    return OUT, IT, GATE, events
