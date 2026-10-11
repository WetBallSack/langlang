"""brooks_sim.py — order fills on 1-minute bars and position management for brooks_core intents.

Rules (spec A9/A10): orders are placed at the close of the 5m signal bar and live during the next bar (stop orders
get one extra bar if the next bar is an inside bar that did not trigger). Stop entries fill only on a trade-through,
at max(stop, 1m open) for gaps; market orders at the next 1m open. Exits: stop-first within a minute, targets as
limits needing a trade-through by one tick, breakeven at +1R, trailing one tick beyond each new confirmed k=2
pivot after a new extreme, entry-bar tightening after a strong entry bar, time stops, flat at the US close (SPY),
and opposite F3/F4 signals that close and reverse. Benchmarks B0/B1 reuse exactly the same exit code.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

import brooks_core as BC

R_STOP, R_TARGET, R_TIME, R_SESSION, R_REVERSE, R_END = 1, 2, 3, 4, 5, 6
TRADE_COLS = ["fam", "side", "sig_bar", "entry_bar", "exit_bar", "fill", "R", "R_gross", "reason", "risk",
              "frA", "riskA_atr", "riskB_atr", "tgtA_R", "tgtB_R", "trA", "trB", "be", "tstop"]
NTC = len(TRADE_COLS)


@njit(cache=True)
def _bar_minutes(b, m_lo, m_hi):
    return m_lo[b], m_hi[b]


@njit(cache=True)
def run_pos(fb, fj, side, fill, raw_fill, frA, stA, tgA, trA, stB, tgB, trB, be, tstop,
            O, H, L, C, SB, SS, SLAST, plo_px, plo_i, phi_px, phi_i, o1, h1, l1, c1, m_lo, m_hi,
            tick, slip, fee, rev_on, IT, allow, use_te):
    """manage a position opened at 5m bar fb (1m index fj, or -1 when filled at the 5m open/level).
    Two tranches (A, B) held in scalars. Returns exit bar, net pnl, gross pnl (both per unit), weighted risk,
    reason, reversal signal bar, reversal family, reversal price, reversal 1m index."""
    n = len(O)
    s = side
    fA = frA
    fB = 1.0 - frA
    oA = fA > 1e-9
    oB = fB > 1e-9
    sA = stA
    sB = stB
    rA = abs(fill - stA)
    rB = abs(fill - stB)
    wrisk = fA * rA + fB * rB
    pnl = 0.0
    gross = 0.0
    fees = fee * fill
    ext = fill
    reason = R_END
    rev_bar = -1
    rev_f = -1
    rev_px = 0.0
    rev_j = -1
    pend_rev = -1
    pend_px = 0.0
    pend_f = -1
    pend_time = False
    eb = n - 1
    done = False
    for bb in range(fb, n):
        a = m_lo[bb]
        z = m_hi[bb]
        if bb == fb and fj >= 0:
            a = fj
        if pend_time:                                    # time stop: market exit at this bar's open
            px = o1[a] if z > a else O[bb]
            if oA:
                pnl += fA * s * (px * (1 - s * slip) - fill); gross += fA * s * (px - raw_fill); fees += fee * fA * px
            if oB:
                pnl += fB * s * (px * (1 - s * slip) - fill); gross += fB * s * (px - raw_fill); fees += fee * fB * px
            oA = False; oB = False
            reason = R_TIME; eb = bb; done = True
            break
        nmin = z - a if z > a else 1
        for mm in range(nmin):
            if z > a:
                j = a + mm
                mo = o1[j]; mh = h1[j]; ml = l1[j]
                isfirst = j == m_lo[bb]
            else:
                j = -1
                mo = O[bb]; mh = H[bb]; ml = L[bb]
                isfirst = True
            if bb == fb and mm == 0 and fj >= 0:
                isfirst = False                          # the fill minute: no gap logic against our own entry
            hitA = oA and ((s > 0 and ml <= sA) or (s < 0 and mh >= sA))
            hitB = oB and ((s > 0 and ml <= sB) or (s < 0 and mh >= sB))
            # reversal stop from an opposite F3/F4 signal of the previous bar: it fills first only if price
            # reaches its level before any protective stop (a falling bar hits the higher level first for a long)
            if pend_rev >= 0 and pend_rev == bb - 1 and ((s > 0 and ml <= pend_px) or (s < 0 and mh >= pend_px)):
                first = (not hitA or s * (pend_px - sA) >= 0) and (not hitB or s * (pend_px - sB) >= 0)
                if first:
                    px = pend_px
                    if isfirst and ((s > 0 and mo < pend_px) or (s < 0 and mo > pend_px)):
                        px = mo
                    if oA:
                        pnl += fA * s * (px * (1 - s * slip) - fill); gross += fA * s * (px - raw_fill); fees += fee * fA * px
                    if oB:
                        pnl += fB * s * (px * (1 - s * slip) - fill); gross += fB * s * (px - raw_fill); fees += fee * fB * px
                    oA = False; oB = False
                    reason = R_REVERSE; eb = bb; rev_bar = bb - 1; rev_px = px; rev_f = pend_f; rev_j = j
                    done = True
                    break
            if hitA:                                     # stop first, then targets
                px = sA
                if isfirst and ((s > 0 and mo < sA) or (s < 0 and mo > sA)):
                    px = mo
                pnl += fA * s * (px * (1 - s * slip) - fill); gross += fA * s * (px - raw_fill); fees += fee * fA * px
                oA = False; reason = R_STOP
            elif oA and ((s > 0 and mh >= tgA + tick) or (s < 0 and ml <= tgA - tick)):
                pnl += fA * s * (tgA - fill); gross += fA * s * (tgA - raw_fill); fees += fee * fA * tgA
                oA = False; reason = R_TARGET
            if hitB:
                px = sB
                if isfirst and ((s > 0 and mo < sB) or (s < 0 and mo > sB)):
                    px = mo
                pnl += fB * s * (px * (1 - s * slip) - fill); gross += fB * s * (px - raw_fill); fees += fee * fB * px
                oB = False; reason = R_STOP
            elif oB and ((s > 0 and mh >= tgB + tick) or (s < 0 and ml <= tgB - tick)):
                pnl += fB * s * (tgB - fill); gross += fB * s * (tgB - raw_fill); fees += fee * fB * tgB
                oB = False; reason = R_TARGET
            if not (oA or oB):
                eb = bb; done = True
                break
        if done:
            break
        # ---- 5m close: management (effective from the next bar)
        newext = (s > 0 and H[bb] > ext) or (s < 0 and L[bb] < ext)
        if newext:
            ext = H[bb] if s > 0 else L[bb]
        if bb == fb:                                                # entry-bar tightening after a strong entry bar
            if s > 0 and SB[bb] > 0:
                lv = L[bb] - tick
                if oA and lv > sA:
                    sA = lv
                if oB and lv > sB:
                    sB = lv
            if s < 0 and SS[bb] > 0:
                lv = H[bb] + tick
                if oA and lv < sA:
                    sA = lv
                if oB and lv < sB:
                    sB = lv
        if be > 0:                                                  # breakeven after +1R of that tranche
            if oA and rA > 0 and s * (ext - fill) >= rA and s * (fill - sA) > 0:
                sA = fill
            if oB and rB > 0 and s * (ext - fill) >= rB and s * (fill - sB) > 0:
                sB = fill
        if newext:                                                  # trail beyond the latest confirmed pivot
            if s > 0 and plo_i[bb] >= fb:
                lv = plo_px[bb] - tick
                if oA and trA > 0 and lv > sA:
                    sA = lv
                if oB and trB > 0 and lv > sB:
                    sB = lv
            if s < 0 and phi_i[bb] >= fb:
                lv = phi_px[bb] + tick
                if oA and trA > 0 and lv < sA:
                    sA = lv
                if oB and trB > 0 and lv < sB:
                    sB = lv
        if SLAST[bb] > 0:                                           # flat at the US close
            px = c1[z - 1] if z > a else C[bb]
            if oA:
                pnl += fA * s * (px * (1 - s * slip) - fill); gross += fA * s * (px - raw_fill); fees += fee * fA * px
            if oB:
                pnl += fB * s * (px * (1 - s * slip) - fill); gross += fB * s * (px - raw_fill); fees += fee * fB * px
            oA = False; oB = False
            reason = R_SESSION; eb = bb; done = True
            break
        if bb - fb + 1 >= tstop:
            pend_time = True
        if rev_on:
            for f in (3, 2):                                        # F4, F3 may close and reverse
                if allow[f] and IT[bb, f, 0] == -s and (use_te == 0 or IT[bb, f, 11] > 0) and IT[bb, f, 1] == BC.T_STOP:
                    pend_rev = bb
                    pend_px = IT[bb, f, 2]
                    pend_f = f
                    break
    if not done:
        px = C[n - 1]
        if oA:
            pnl += fA * s * (px - fill); gross += fA * s * (px - raw_fill)
        if oB:
            pnl += fB * s * (px - fill); gross += fB * s * (px - raw_fill)
        reason = R_END; eb = n - 1
    return eb, pnl - fees, gross, wrisk, reason, rev_bar, rev_f, rev_px, rev_j


@njit(cache=True)
def try_fill(b, etype, px, side, O, H, L, INS, SLAST, o1, h1, l1, m_lo, m_hi, tick, slip):
    """order placed at the close of bar b. Returns (filled, bar, 1m index or -1, fill with slippage, raw fill)"""
    n = len(O)
    last = b + 1
    if etype == BC.T_STOP and b + 2 < n and INS[b + 1] > 0 and SLAST[b + 1] == 0:
        last = b + 2
    for bb in range(b + 1, min(last, n - 1) + 1):
        a, z = m_lo[bb], m_hi[bb]
        if etype == BC.T_MARKET:
            raw = o1[a] if z > a else O[bb]
            return True, bb, (a if z > a else -1), raw * (1 + side * slip), raw
        if z > a:
            for j in range(a, z):
                if etype == BC.T_STOP and ((side > 0 and h1[j] >= px) or (side < 0 and l1[j] <= px)):
                    raw = max(px, o1[j]) if side > 0 else min(px, o1[j])
                    return True, bb, j, raw * (1 + side * slip), raw
                if etype == BC.T_LIMIT and ((side > 0 and l1[j] < px - tick) or (side < 0 and h1[j] > px + tick)):
                    return True, bb, j, px, px
        else:
            if etype == BC.T_STOP and ((side > 0 and H[bb] >= px) or (side < 0 and L[bb] <= px)):
                raw = max(px, O[bb]) if side > 0 else min(px, O[bb])
                return True, bb, -1, raw * (1 + side * slip), raw
            if etype == BC.T_LIMIT and ((side > 0 and L[bb] < px - tick) or (side < 0 and H[bb] > px + tick)):
                return True, bb, -1, px, px
        if SLAST[bb] > 0:
            break
    return False, -1, -1, 0.0, 0.0


@njit(cache=True)
def simulate(IT, allow, use_te, scalp_only, O, H, L, C, ATR, SB, SS, INS, SLAST, plo_px, plo_i, phi_px, phi_i,
             o1, h1, l1, c1, m_lo, m_hi, tick, slip, fee, rev_on, out):
    n = len(O)
    nt = 0
    b = 0
    rv_f = -1
    rv_bar = -1
    rv_px = 0.0
    rv_exit = -1
    rv_j = -1
    while b < n - 2:
        f = -1
        if rv_f >= 0:                                              # reversal: already filled at rv_px on bar rv_exit
            f = rv_f
            sig = rv_bar
        else:
            if SLAST[b] > 0:
                b += 1
                continue
            for k in range(6):
                ff = BC.PRIORITY[k]
                if allow[ff] and IT[b, ff, 0] != 0 and (use_te == 0 or IT[b, ff, 11] > 0):
                    f = ff
                    break
            if f < 0:
                b += 1
                continue
            sig = b
        side = int(IT[sig, f, 0])
        if rv_f >= 0:
            ok, fb, fj, fill, raw = True, rv_exit, rv_j, rv_px * (1 + side * slip), rv_px
            rv_f = -1
        else:
            ok, fb, fj, fill, raw = try_fill(sig, int(IT[sig, f, 1]), IT[sig, f, 2], side, O, H, L, INS, SLAST,
                                             o1, h1, l1, m_lo, m_hi, tick, slip)
        if not ok:
            b += 1
            continue
        frA = IT[sig, f, 7]
        stA, stB, tgA, tgB = IT[sig, f, 3], IT[sig, f, 4], IT[sig, f, 5], IT[sig, f, 6]
        trB = IT[sig, f, 8]
        trA = trB if frA >= 0.999 else 0.0
        if scalp_only:
            frA = 1.0; stB = stA; trA = 0.0; trB = 0.0
            tgA = raw + side * abs(raw - stA); tgB = tgA
        if (fill - stA) * side <= 0 or (frA < 0.999 and (fill - stB) * side <= 0):
            b = fb                                               # gapped beyond the protective stop: skip
            continue
        eb, pnl, gross, wrisk, why, rbar, rf, rpx, rj = run_pos(
            fb, fj, side, fill, raw, frA, stA, tgA, trA, stB, tgB, trB, IT[sig, f, 12], int(IT[sig, f, 9]),
            O, H, L, C, SB, SS, SLAST, plo_px, plo_i, phi_px, phi_i, o1, h1, l1, c1, m_lo, m_hi,
            tick, slip, fee, rev_on, IT, allow, use_te)
        if nt < out.shape[0] and wrisk > 0:
            a = ATR[sig] if ATR[sig] > 0 else 1.0
            out[nt, 0] = f; out[nt, 1] = side; out[nt, 2] = sig; out[nt, 3] = fb; out[nt, 4] = eb; out[nt, 5] = fill
            out[nt, 6] = pnl / wrisk; out[nt, 7] = gross / wrisk; out[nt, 8] = why; out[nt, 9] = wrisk
            out[nt, 10] = frA; out[nt, 11] = abs(fill - stA) / a; out[nt, 12] = abs(fill - stB) / a
            out[nt, 13] = abs(tgA - fill) / max(abs(fill - stA), 1e-12)
            out[nt, 14] = abs(tgB - fill) / max(abs(fill - stB), 1e-12)
            out[nt, 15] = trA; out[nt, 16] = trB; out[nt, 17] = IT[sig, f, 12]; out[nt, 18] = IT[sig, f, 9]
            nt += 1
        if why == R_REVERSE and rf >= 0:
            rv_f = rf; rv_bar = rbar; rv_px = rpx; rv_exit = eb; rv_j = rj
            b = eb
            continue
        b = max(eb, b + 1)
    return nt


@njit(cache=True, parallel=True)
def bench(reps, seed, tr, pool_lo, pool_hi, pool_items, tr_group, O, H, L, C, ATR, SB, SS, SLAST, plo_px, plo_i, phi_px,
          phi_i, o1, h1, l1, c1, m_lo, m_hi, tick, slip, fee):
    """B0/B1: for each real trade, one random signal bar from its pool; market entry at the next bar's open; the
    same stop distances (ATR units), targets (R multiples), fractions, trailing, breakeven and time stop."""
    ntr = tr.shape[0]
    res = np.zeros(reps)
    n = len(O)
    dummy_it = np.zeros((1, 6, BC.NI))
    allow = np.zeros(6, np.bool_)
    for r in prange(reps):
        st = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(r + 1) * np.uint64(1442695040888963407)
        tot = 0.0
        cnt = 0
        for q in range(ntr):
            g = tr_group[q]
            a0, a1 = pool_lo[g], pool_hi[g]
            if a1 <= a0:
                continue
            st ^= st << np.uint64(13)
            st ^= st >> np.uint64(7)
            st ^= st << np.uint64(17)
            b = pool_items[a0 + int(st % np.uint64(a1 - a0))]
            if b + 2 >= n:
                continue
            fb = b + 1
            fj = m_lo[fb] if m_hi[fb] > m_lo[fb] else -1
            raw = o1[fj] if fj >= 0 else O[fb]
            s = int(tr[q, 1])
            fill = raw * (1 + s * slip)
            a = ATR[b]
            if not (a > 0):
                continue
            rA = tr[q, 11] * a
            rB = tr[q, 12] * a
            stA = fill - s * rA
            stB = fill - s * rB
            tgA = fill + s * tr[q, 13] * rA
            tgB = fill + s * tr[q, 14] * rB
            eb, pnl, gross, wrisk, why, rb, rf, rp, rj = run_pos(
                fb, fj, s, fill, raw, tr[q, 10], stA, tgA, tr[q, 15], stB, tgB, tr[q, 16], tr[q, 17], int(tr[q, 18]),
                O, H, L, C, SB, SS, SLAST, plo_px, plo_i, phi_px, phi_i, o1, h1, l1, c1, m_lo, m_hi,
                tick, slip, fee, False, dummy_it, allow, 0)
            if wrisk > 0:
                tot += pnl / wrisk
                cnt += 1
        res[r] = tot / cnt if cnt else 0.0
    return res


def pivot_tracks(FE):
    """last CONFIRMED k=2 pivot low/high (price, pivot bar index) known at each bar close"""
    n = FE.shape[0]
    plo_px = np.full(n, np.nan); plo_i = np.full(n, -1, np.int64)
    phi_px = np.full(n, np.nan); phi_i = np.full(n, -1, np.int64)
    pl = np.flatnonzero(FE[:, BC.C_PL2] > 0)
    ph = np.flatnonzero(FE[:, BC.C_PH2] > 0)
    for arr, px_out, i_out, col in ((pl, plo_px, plo_i, BC.C_L), (ph, phi_px, phi_i, BC.C_H)):
        conf = arr + 2
        ok = conf < n
        mark = np.full(n, -1, np.int64)
        mark[conf[ok]] = arr[ok]
        last = -1
        for t in range(n):
            if mark[t] >= 0:
                last = mark[t]
            if last >= 0:
                i_out[t] = last
                px_out[t] = FE[last, col]
    return plo_px, plo_i, phi_px, phi_i


def minute_map(t5, t1, step_ns=300_000_000_000):
    """[m_lo, m_hi) 1m indices inside each 5m bar"""
    if t1 is None or len(t1) == 0:
        z = np.zeros(len(t5), np.int64)
        return z, z
    m_lo = np.searchsorted(t1, t5)
    m_hi = np.searchsorted(t1, t5 + step_ns)
    return m_lo.astype(np.int64), m_hi.astype(np.int64)
