# M1: why CFG does not take off and critic-argmax does (take-off regime, seed 0, models after 5 rounds)
At 48 states reached after 5 decisions on in-air goals (0.10–0.30 m), sample 32 candidate chunks from each
method's generator. "up" is the mean z-displacement command of a candidate chunk. `rsi/analysis_direction.py`.

| improvement signal | Spearman(score, up) | up of top-scored minus median | effect of guidance on mean up |
|---|---|---|---|
| CFG implicit ratio log μ(a\|s,succ) − log μ(a\|s) | −0.077 | −0.022 | w = 0 / 1 / 2 / 3: −0.0153 / −0.0175 / −0.0191 / −0.0204 (pushes **down**) |
| critic Q (argmax K = 64, filtered init) | **+0.373** | **+0.114** | n/a |

Reading:
- The conditional generator learns what successful actions look like. Every success it has seen (demos,
  hindsight relabels) is at table height. For an in-air goal it falls back to the nearest supported
  behaviour, so guidance sharpens toward keeping the object on the table: it interpolates within the
  support.
- The critic is a discriminative model of P(success | s, a, g). Hindsight relabels teach it that success
  means the object reaching g. That is a smooth, monotone relation, so it extrapolates: for higher g,
  candidates that move up score higher. Argmax over many candidates turns this extrapolation into
  exploration.
- This is the robot instance of the LLM result that sharpening (SFT/CFG-style) cannot exceed the base
  policy's coverage; going beyond needs an exploration signal (here, the critic's extrapolation plus large K).
- Consequence for the method search: keep the critic for direction (take-off). Generative guidance is only
  a within-support tool.
