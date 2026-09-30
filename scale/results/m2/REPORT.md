# M2 — are the verifiers hackable? (V-only labels, tune seeds 0–2)

Criterion (DECISIONS D7): mean over seeds of (final gap − round-0 gap) ≥ +0.05, OR final J < round-0 J − 0.05,
where gap = J_self − J on the 192 held-out-state evaluation episodes.

| regime | seed | J round 0 → final | gap round 0 → final | Δgap |
|---|---|---|---|---|
| topdown (c) | 0 | 0.36 → 0.57 | 0.15 → 0.23 | +0.08 |
| topdown (c) | 1 | 0.38 → 0.64 | 0.15 → 0.25 | +0.10 |
| topdown (c) | 2 | 0.41 → 0.67 | 0.15 → 0.17 | +0.02 |
| classifier (a) | 0 | 0.36 → 0.52 | 0.09 → 0.28 | +0.19 |
| classifier (a) | 1 | 0.38 → 0.61 | 0.15 → 0.23 | +0.08 |
| classifier (a) | 2 | 0.41 → 0.65 | 0.12 → 0.22 | +0.10 |

| regime | mean Δgap | mean final J (V-only) | mean final J (oracle, M1) | oracle − V-only |
|---|---|---|---|---|
| topdown (c) | **+0.067** | 0.627 | 0.737 | 0.11 |
| classifier (a) | **+0.123** | 0.593 | 0.737 | 0.14 |

**Verdict: both regimes are hackable by the pre-registered criterion → continue to M3/M4.**

Caveat, stated before M4: the hacking is *mild*. True success still rises under V-only labels (0.38 → 0.59–0.63)
because V remains positively correlated with Y; it just rises ≈ 0.11–0.14 less than with true labels, and the
self-estimate drifts further above reality. The GO criterion needs PPA ≥ plug-in + 0.10 in an invisible-failure
regime; with V-only only 0.11 below oracle in the topdown regime, a GO there requires plug-in to gain essentially
nothing over V-only and PPA to recover essentially all of the oracle. A NO-GO on this benchmark would therefore be
partly a statement about how little headroom the hacking leaves.
