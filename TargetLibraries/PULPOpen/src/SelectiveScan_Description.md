# `PULP_SelectiveScan_i8_i8` — Kernel Description

A precise, line-referenced walkthrough of the quantized **Mamba-style selective
scan (S6) with SiLU gating** implemented in
[SelectiveScan.c](SelectiveScan.c). The goal of this document is to make the
binding between the on-target integer kernel and the **state-space-model (SSM)
selective-scan algorithm** explicit at every step.

This kernel mirrors, bit-for-bit, the quantized selective-scan Python reference
used to build its test vectors; the exp LUT (all 2561 Q15 entries) lives in
[inc/kernel/SelectiveScanLUT.h](../inc/kernel/SelectiveScanLUT.h); the call is emitted
by `Deeploy/Targets/PULPOpen/Templates/SelectiveScanTemplate.py` and the tensor
roles / gate LUT are set up in `PULPSelectiveScanParser`
(`Deeploy/Targets/PULPOpen/Parsers.py`).

This PULPOpen (Siracusa) kernel and the GAP9 kernel
(`TargetLibraries/GAP9/src/SelectiveScan.c`) are **numerically identical**
(verified bit-exact); they differ only in target-specific details (see §7). The
performance instrumentation has been moved out of the production source into
[SelectiveScan_Instrumented.md](SelectiveScan_Instrumented.md).

---

## 1. The SOTA algorithm being implemented

### 1.1 Continuous SSM
A diagonal state-space model over a scalar input channel is

$$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t) + D\,x(t)$$

with per-channel diagonal state $h \in \mathbb{R}^{N}$, state matrix
$A \in \mathbb{R}^{N}$ (diagonal, **negative** for stability), and input/output
projections $B, C \in \mathbb{R}^{N}$.

### 1.2 Selective (input-dependent) discretization — Mamba S6
Mamba makes $\Delta$, $B$, $C$ **functions of the input** (selectivity) and
discretizes with a **zero-order hold** on $A$ and the **simplified Euler** rule
on $B$ (exactly the rule used by the reference `selective_scan` CUDA kernel):

$$\bar{A}_t = \exp(\Delta_t A), \qquad \bar{B}_t = \Delta_t B_t$$

giving the **linear recurrence (the "scan")** and the output read-out:

$$\boxed{\;h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t\, x_t\;}
\qquad
\boxed{\;y_t = \sum_{n} C_{t}\, h_t + D\, x_t\;}$$

### 1.3 Gating (Mamba block tail)
The SSM output is multiplied by a **SiLU-activated gate branch** $z$:

$$y_t \leftarrow y_t \cdot \operatorname{SiLU}(z_t), \qquad
\operatorname{SiLU}(z) = z\,\sigma(z)$$

This kernel fuses *all three* stages — discretization, recurrence + read-out,
and SiLU gating — plus the final requantization to `int8`.

### 1.4 Index conventions in this kernel
* `b` — batch element, `b ∈ [0, B_size)`
* `t` — sequence position (time), `t ∈ [0, L)`
* `d` — inner channel, `d ∈ [0, D_inner)`  (Mamba's `d_inner`)
* `n` — diagonal state index, `n ∈ [0, N)`  (Mamba's `d_state`, here `N ≤ 16`)

Selectivity layout (matches Mamba S6): $\Delta$ is per `(b,t,d)`; $B$ and $C$ are
per `(b,t,n)` (**shared across channels `d`**); $A$ is per `(d,n)` and $D$ per
`d` (both static, time-invariant).

---

## 2. Signature and tensor semantics

Kernel signature: [SelectiveScan.c:40-47](SelectiveScan.c#L40-L47).

| Arg | C type / shape | Q-format | SSM role |
|-----|----------------|----------|----------|
| `x` | `int8 [B,L,D_inner]` | integer | SSM input $x$ (Mamba `u`, post-conv/SiLU x-branch) |
| `z` | `int8 [B,L,D_inner]` | integer index | **gate** branch $z$ (pre-SiLU); used as LUT index |
| `dt` | `int16 [B,L,D_inner]` | **Q8.8** | selective timestep $\Delta$ (softplus output, ≥ 0) |
| `B` | `int32 [B,L,N]` | wide (scale folded upstream) | selective input projection $B_t$ |
| `C` | `int32 [B,L,N]` | wide (scale folded upstream) | selective output projection $C_t$ |
| `A` | `int32 [D_inner,N]` | **Q15** | diagonal state matrix $A$ (negative) |
| `D_skip` | `int32 [D_inner]` | wide | skip / residual term $D$ |
| `y` | `int8 [B,L,D_inner]` | integer | gated, requantized output |
| `h_buffer` | `int32 [B,D_inner,N]` | wide state domain | **persistent** hidden state $h$ (streamed across L-tiles) |
| `gate_lut` | `int32 [256]` | **Q13** | precomputed $\operatorname{SiLU}(z)$ table, index `z+128` |
| `B_size,L,D_inner,N` | `uint32` | — | dimensions |
| `output_requant_mul_q40` | `int32` | **Q40** | composite wide→int8 output rescale |
| `is_first_L_tile` | `uint32` | flag | if set, zero `h` before the scan |

The `gate_lut` is built at Deeploy compile time in `PULPSelectiveScanParser`
(`Parsers.py:582-587`): `q20 = round(z·σ(z)·2^20)` then rounded to **Q13**, so
`gate_lut[z+128] = round(SiLU(z)·2^13)`. `output_requant_mul_q40` is
`round(x_scale · 2^(40−15) / out_scale)` — it folds the input scale, the
Q15 wide-domain de-scale, and the output scale into one Q40 multiplier.

---

## 3. Fixed-point domains and constants

Defined at [SelectiveScan.c:12-15](SelectiveScan.c#L12-L15):

| Constant | Value | Meaning |
|----------|-------|---------|
| `SSM_WIDE_FRAC_BITS` | 15 | Q15 fractional bits for $\bar A$, the state gain, and read-out |
| `SSM_EXP_RANGE_Q15` | `20 << 15` = 655360 | exp argument clipped to $[-20, 0]$ (in Q15) |
| `SSM_EXP_HALF_Q15` | 128 | half-step, for round-to-nearest LUT indexing |
| `SSM_EXP_STEP_LOG2_Q15` | 8 | LUT step = `2^8` = 256 in Q15 (= 1/128 in real units) |

The exp LUT (`SelectiveScan_exp_lut_qwide`,
[SelectiveScanLUT.h:18-252](../inc/kernel/SelectiveScanLUT.h#L18-L252)) has
`2561 = 655360/256 + 1` entries tabulating
$\bar A = \exp(\Delta A)$ in Q15 over $\Delta A \in [-20, 0]$ at 1/128 spacing.
The first 1137 entries are `0` (`exp(-20)·2^15 ≈ 0`); the last is `32767 ≈ 1.0`
in Q15 (`exp(0)=1`).

**Domain chain for one state lane** (`dt` is Q8.8, `A` is Q15):
$$\Delta A \big|_{Q15} = (\underbrace{dt}_{Q8.8}\cdot \underbrace{A}_{Q15}) \gg 8,
\qquad
\bar A \big|_{Q15} = \mathrm{LUT}[\,\cdot\,],
\qquad
h \leftarrow (\bar A \cdot h)\gg 15 \;+\; (\Delta B)\,x .$$

The `>> 8` strips $\Delta$'s 8 fractional bits; the `>> 15` strips $\bar A$'s
Q15 gain. `B`/`C` carry the residual scale folded from the upstream projections.

---

## 4. Helper functions

* **`_ssm_sat_i32_i64`** ([18-24](SelectiveScan.c#L18-L24)) — branch-predicted
  saturating narrow of the `int64` state accumulator to `int32`. Fast path:
  `hi == (lo >> 31)` means the value fits in 32 bits; otherwise clamp to
  `INT32_MAX/MIN`. Keeps the hidden state in `int32`.
* **`_ssm_round_shift_i64`** ([27-32](SelectiveScan.c#L27-L32)) — symmetric
  (round-half-away-from-zero, sign-preserving) right shift. Used for the two
  final requant multiplies so positive and negative outputs round identically.
* **`_ssm_exp_lut_lookup`** ([36-38](SelectiveScan.c#L36-L38)) — a
  `noinline` wrapper around a single LUT read. It is deliberately *not* inlined:
  the machine-level `jalr` it emits inside the `n`-loop blocks the XpulpV2
  hardware-loop pass, which otherwise mis-compiles the loop and silently drops
  h-state stores (see §7).

---

## 5. Step-by-step kernel flow

### 5.1 Core work partition over `D_inner`
[SelectiveScan.c:51-54](SelectiveScan.c#L51-L54). The 8-core PULP cluster
splits the channel axis: each core owns a contiguous chunk
`[D_start, D_end)` of `D_inner`. Work is partitioned over `d` (not `t`) because
the scan is **sequential in `t`** but **fully independent across `d`** — each
channel has its own private recurrence.

```c
D_chunk = ceil(D_inner / NUM_CORES);
D_start = min(core_id * D_chunk, D_inner);
D_end = min(D_start + D_chunk, D_inner);
```

### 5.2 State (re)initialization — the L-tiling streaming contract
[SelectiveScan.c:57-66](SelectiveScan.c#L57-L66). `h_buffer` holds
$h_{t-1}$ *across kernel invocations*. When `is_first_L_tile` is set (start of a
new sequence), each core zeros its own `[B_size, D_start:D_end, N]` slice
($h_{-1}=0$), then a `pi_cl_team_barrier()` ensures all state is cleared before
the scan reads it. On later L-tiles the buffer is **kept**, so the recurrence
continues seamlessly from where the previous tile stopped — this is what makes
sequence tiling correct. Cores with no work (`D_start >= D_end`, possible when
`D_inner < NUM_CORES`) still hit the barrier and return
([68-71](SelectiveScan.c#L68-L71)).

### 5.3 Per-channel hoisting `(b, d)`
Loop order is `(b, d, t, n)` — [SelectiveScan.c:74-75](SelectiveScan.c#L74-L75).
For each `(b,d)`:
* **`A_local[16]`** ([78-83](SelectiveScan.c#L78-L83)) — the state row
  `A[d,:]` is time-invariant, so it is copied to the stack once and reused for
  all `L` steps (removes reload + pointer aliasing in the hot loop).
* **`h_local[16]`** ([87-90](SelectiveScan.c#L87-L90)) — the persistent
  state row is loaded from `h_buffer` into a stack array. The whole `t`-scan
  runs on `h_local` (no aliasing, register-friendly) and is written back once
  at the end (§5.7).
* **`d_skip_val`** ([92-94](SelectiveScan.c#L92-L94)) — `D_skip[d]` is
  `t`-invariant and is hoisted out of the `t`-loop, saving one L1 load per step.

### 5.4 Time step — gather selective inputs
For each `t`, [SelectiveScan.c:98-103](SelectiveScan.c#L98-L103) gathers the
inputs of time step `t`:
`B_row = B[b,t,:]`, `C_row = C[b,t,:]` (state-indexed, shared across `d`),
and the scalars `x_val = x[b,t,d]`, `z_val = z[b,t,d]`, `dt_val = dt[b,t,d]`.
`y_acc` (the `int64` read-out accumulator $\sum_n C h$) is reset to 0.

### 5.5 The state recurrence — inner `n`-loop (the heart of the scan)
[SelectiveScan.c:105-126](SelectiveScan.c#L105-L126). For each state lane `n`
this computes one term of $h_t = \bar A \odot h_{t-1} + \bar B x$ and
accumulates one term of $y_t = \sum_n C\,h_t$. The LUT-independent $\bar B$ path
is computed **first** so its multiplies overlap the latency of the
`_ssm_exp_lut_lookup` call on the in-order pipeline:

| Line | Code | SSM meaning |
|------|------|-------------|
| [108](SelectiveScan.c#L108) | `dB_q15 = (dt_val * B_row[n]) >> 8` | $\bar B_t = \Delta_t B_{t,n}$ (simplified Euler) |
| [109](SelectiveScan.c#L109) | `dB_x_q15 = dB_q15 * x_val` | $\bar B_t\, x_t$ (input drive; computed early to hide LUT latency) |
| [111](SelectiveScan.c#L111) | `dt_A_q15 = (dt_val * A_local[n]) >> 8` | $\Delta_t \cdot A_{d,n}$ then $\gg 8$ → $\Delta A$ in Q15 (Q8.8·Q15) |
| [112-115](SelectiveScan.c#L112-L115) | clip to `[-RANGE, 0]` | keep exp arg in $[-20,0]$; $\Delta A>0 \Rightarrow 0$ (so $\bar A=1$) |
| [116-119](SelectiveScan.c#L116-L119) | `exp_idx = (clip + RANGE + HALF) >> 8` | round-to-nearest LUT index |
| [120](SelectiveScan.c#L120) | `dA_q15 = _ssm_exp_lut_lookup(exp_idx)` | $\bar A_t = \exp(\Delta A)$ in Q15 |
| [121](SelectiveScan.c#L121) | `h_acc = (dA_q15 * h_local[n]) >> 15` | $\bar A_t \odot h_{t-1}$ (de-scale Q15 gain) |
| [122](SelectiveScan.c#L122) | `h_acc += dB_x_q15` | $+\,\bar B_t x_t$ → completes $h_t$ |
| [123](SelectiveScan.c#L123) | `h_new = _ssm_sat_i32_i64(h_acc)` | saturate state to int32 |
| [124](SelectiveScan.c#L124) | `y_acc += (h_new * C_row[n]) >> 15` | $+\,C_{t,n}\,h_t$ (read-out term) |
| [125](SelectiveScan.c#L125) | `h_local[n] = h_new` | commit $h_t$ for the next time step |

Note the ordering: `h_local[n]` is updated **before** it feeds `y_acc`, so the
read-out uses the *post-update* state $h_t$ (Mamba's convention
$y_t = C_t h_t$), not $h_{t-1}$.

### 5.6 Skip term, gating, and requantization
After the `n`-loop, still inside the `t`-loop:

1. **Skip / residual $D x$** — [128](SelectiveScan.c#L128):
   `y_acc += d_skip_val * x_val`, completing $y_t = \sum_n C h_t + D x_t$.
2. **SiLU gate** — [129-130](SelectiveScan.c#L129-L130):
   `gate_q13 = gate_lut[z_val + 128]` reads $\operatorname{SiLU}(z_t)$ in Q13;
   `y_gated = round_shift(y_acc * gate_q13, 13)` applies
   $y_t \cdot \operatorname{SiLU}(z_t)$ and strips the Q13 scale. The `z+128`
   offset maps the signed `int8` gate value into the `[0,255]` table.
3. **Output requant** — [131-132](SelectiveScan.c#L131-L132):
   `y_out = round_shift(y_gated * output_requant_mul_q40, 40)` rescales the wide
   accumulator to the int8 output scale (Q40 multiply + rounding `>>40`).
4. **Saturate & store** — [133-137](SelectiveScan.c#L133-L137): clamp to
   `[-128, 127]` and write `y[b,t,d]`.

### 5.7 Persist state for the next L-tile
[SelectiveScan.c:140-142](SelectiveScan.c#L140-L142). After the full `t`-scan
for `(b,d)`, `h_local` is written back to `h_buffer` so a subsequent L-tile (or
the next call) resumes the recurrence from $h_{L-1}$ (see §5.2).

### 5.8 Final synchronization
[SelectiveScan.c:147](SelectiveScan.c#L147). A closing `pi_cl_team_barrier()`
joins all cores — both the mandatory Deeploy inter-kernel sync point and a
guarantee that every core's `y` / `h_buffer` writes are committed.

---

## 6. Consolidated mapping: SOTA ⇄ kernel

| SSM / Mamba S6 | Discretized form | Kernel expression | Location |
|----------------|------------------|-------------------|----------|
| $\Delta_t$ (selective step) | softplus, Q8.8 | `dt_val` | [102](SelectiveScan.c#L102) |
| $A_{d,n}$ (diag, <0) | static, Q15 | `A_local[n]` | [111](SelectiveScan.c#L111) |
| $\bar A_t = \exp(\Delta_t A)$ | ZOH, Q15 LUT | `dA_q15` | [111-120](SelectiveScan.c#L111-L120) |
| $\bar B_t = \Delta_t B_t$ | simplified Euler | `dB_q15` | [108](SelectiveScan.c#L108) |
| $x_t$ | input | `x_val` | [100](SelectiveScan.c#L100) |
| $h_t = \bar A_t h_{t-1} + \bar B_t x_t$ | recurrence | `h_acc → h_local[n]` | [121-125](SelectiveScan.c#L121-L125) |
| $C_t$ | selective read-out | `C_row[n]` | [124](SelectiveScan.c#L124) |
| $\sum_n C_t h_t$ | read-out | `y_acc` | [124](SelectiveScan.c#L124) |
| $+\,D_d x_t$ | skip | `d_skip_val*x_val` | [128](SelectiveScan.c#L128) |
| $\operatorname{SiLU}(z_t)$ | gate | `gate_lut[z_val+128]` | [129](SelectiveScan.c#L129) |
| $y_t \cdot \operatorname{SiLU}(z_t)$ | gating | `y_gated` | [130](SelectiveScan.c#L130) |
| output requant → int8 | Q40 rescale | `y_out` | [131-137](SelectiveScan.c#L131-L137) |

---

## 7. Implementation notes (why the code looks the way it does)

* **`noinline` exp LUT lookup.** The `jalr` emitted by `_ssm_exp_lut_lookup`
  inside the `n`-loop is a deliberate barrier to the Clang 15 / rv32imf_xpulpv2
  hardware-loop (`lp.setup`) pass. That pass software-pipelines the loop and
  places continuation-block stores at the wrong position, **silently dropping
  h-state updates** — a correctness bug. The far call prevents the conversion.
  See the header comment at [34-38](SelectiveScan.c#L34-L38). This is the main
  divergence from the GAP9 sibling, which reads the LUT directly.
* **`dB` computed before the LUT.** The $\bar B$ multiplies
  ([108-109](SelectiveScan.c#L108-L109)) depend only on `dt_val`, `B_row[n]` and
  `x_val`, not on the LUT chain, so they are scheduled ahead of the LUT call to
  overlap its latency on the in-order pipeline. Bit-identical to computing them
  after.
* **Stack-hoisted `A_local` / `h_local` / `d_skip_val`.** `int32[16]` fixed-size
  arrays remove pointer aliasing (`__restrict__` alone is not enough across the
  LUT call) and let the compiler keep the state row in registers; the state is
  loaded/stored from `h_buffer` exactly once per `(b,d)`, and `D_skip[d]` is
  read once per `(b,d)` instead of once per `t`.
* **`int64` accumulators.** `h_acc` and `y_acc` are 64-bit to hold the wide
  products before the Q15 de-scale and the final saturating narrow.
* **Performance instrumentation.** The production kernel is de-instrumented; the
  `perf_bench_*` whole-kernel profiler is preserved (commented) in
  [SelectiveScan_Instrumented.md](SelectiveScan_Instrumented.md) together with a
  description of what it measures. The current shape (plain `n`-loop +
  `noinline` LUT) was found optimal by a Pareto study at ~200.9 cycles/D-step,
  IPC 0.652 — load-stall bound at the LUT-call ABI boundary.
