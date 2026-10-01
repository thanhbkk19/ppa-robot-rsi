# A1: where the round-0 gain of CFG comes from (demo outcome labels)
Conditional BC (CFG) is trained on demos labelled success / failure. Every earlier pipeline trained BC on
all demos without labels. To separate "guidance" from "using the labels", compare with filtered BC:
unconditional diffusion on the successful demos only, with the same demos and the same 15 000 steps.

| regime | BC (all demos) | filtered BC | CFG w=1 | CFG w=2 | CFG w=2, v=1 | RSI argmax K=2 after 6 rounds |
|---|---|---|---|---|---|---|
| easy (3 seeds) | 0.48 | **0.94** | 0.84 | 0.94 | 0.96 | 0.895 |
| hard (3 seeds) | 0.09 | **0.93** | 0.52 | 0.71 | 0.80 | 0.753 |

Findings:
1. The round-0 gain of CFG is explained by the demo labels. Filtered BC matches CFG in the easy regime and
   beats it in the hard one (0.93 vs 0.80). In the hard regime the conditional model fits the
   success-conditioned distribution worse than a model trained only on the successful rows (61% of rows
   are failures).
2. Confound in the earlier in-distribution study (D1b, tuning, RB, lcbopt): their headroom came mostly from
   demo-quality heterogeneity, which filtering removes for free (≈ 0.93 before any online episode).
   The K-inversion and step-size findings still hold as statements about the loop, but those regimes do
   not test RSI beyond what labelled demos already give.
3. The take-off regime is unaffected: no demo has an in-air goal, so filtering cannot help (CFG at round 0
   is 0.00 for every w). From here on, take-off is the main testbed, and every baseline starts from
   filtered BC so that all methods have the same information.
