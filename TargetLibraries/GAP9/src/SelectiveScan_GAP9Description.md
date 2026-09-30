# `GAP9_SelectiveScan_i8_i8` — Kernel Description (GAP9)

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

Kernel signature: [SelectiveScan.c:54-61](SelectiveScan.c#L54-L61).

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

The `gate_lut` is built at Deeploy compile time in `PULPSelectiveScanParser`:
`q20 = round(z·σ(z)·2^20)` then rounded to **Q13**, so
`gate_lut[z+128] = round(SiLU(z)·2^13)`. `output_requant_mul_q40` is
`round(x_scale · 2^(40−15) / out_scale)` — it folds the input scale, the
Q15 wide-domain de-scale, and the output scale into one Q40 multiplier.

---

## 3. Fixed-point domains and constants

Defined at [SelectiveScan.c:13-16](SelectiveScan.c#L13-L16):

| Constant | Value | Meaning |
|----------|-------|---------|
| `SSM_WIDE_FRAC_BITS` | 15 | Q15 fractional bits for $\bar A$, the state gain, and read-out |
| `SSM_EXP_RANGE_Q15` | `20 << 15` = 655360 | exp argument clipped to $[-20, 0]$ (in Q15) |
| `SSM_EXP_HALF_Q15` | 128 | half-step, for round-to-nearest LUT indexing |
| `SSM_EXP_STEP_LOG2_Q15` | 8 | LUT step = `2^8` = 256 in Q15 (= 1/128 in real units) |

The exp LUT (`SelectiveScan_exp_lut_qwide`,
[PULPOpen SelectiveScanLUT.h:18-252](../../PULPOpen/inc/kernel/SelectiveScanLUT.h#L18-L252)) has
`2561 = 655360/256 + 1` entries tabulating
$\bar A = \exp(\Delta A)$ in Q15 over $\Delta A \in [-20, 0]$ at 1/128 spacing.
The first 1137 entries are `0` (`exp(-20)·2^15 ≈ 0`); the last is `32767 ≈ 1.0`
in Q15 (`exp(0)=1`). It is declared `static PI_L1` in `SelectiveScanLUT.h`
(included **directly** by both `.c` files, not via any umbrella header), so it
lives in L1 and is read directly by the hot loop.

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

* **`_ssm_sat_i32_i64`** ([22-28](SelectiveScan.c#L22-L28)) — branch-predicted
  saturating narrow of the `int64` state accumulator to `int32`. Fast path:
  `hi == (lo >> 31)` means the value fits in 32 bits; otherwise clamp to
  `INT32_MAX/MIN`. Keeps the hidden state in `int32`.
* **`_ssm_round_shift_i64`** ([31-36](SelectiveScan.c#L31-L36)) — symmetric
  (round-half-away-from-zero, sign-preserving) right shift. Used for the two
  final requant multiplies so positive and negative outputs round identically.
* **`_ssm_h_update`** ([38-48](SelectiveScan.c#L38-L48)) — `static inline`; the
  **whole per-lane state update**: $\bar B$ path, exp-LUT read, $\bar A \odot h$,
  and the saturating narrow. This is where the GAP9 kernel reads the LUT
  **directly** (`SelectiveScan_exp_lut_qwide[exp_idx]`, [44](SelectiveScan.c#L44)),
  unlike the PULPOpen sibling.
* **`_ssm_y_contrib`** ([50-52](SelectiveScan.c#L50-L52)) — `static inline`; one
  read-out term $\big(C_{t,n}\,h_t\big)\gg 15$.

---

## 5. Step-by-step kernel flow

### 5.1 Core work partition over `D_inner`
[SelectiveScan.c:66-68](SelectiveScan.c#L66-L68). The 9-core GAP9 cluster splits
the channel axis: each core owns a contiguous chunk `[D_start, D_end)` of
`D_inner`. Work is partitioned over `d` (not `t`) because the scan is
**sequential in `t`** but **fully independent across `d`** — each channel has its
own private recurrence.

```c
D_chunk = ceil(D_inner / NUM_CORES);
D_start = min(core_id * D_chunk, D_inner);
D_end = min(D_start + D_chunk, D_inner);
```

### 5.2 State (re)initialization — the L-tiling streaming contract
[SelectiveScan.c:71-80](SelectiveScan.c#L71-L80). `h_buffer` holds $h_{t-1}$
*across kernel invocations*. When `is_first_L_tile` is set (start of a new
sequence), each core zeros its own `[B_size, D_start:D_end, N]` slice
($h_{-1}=0$), then a `pi_cl_team_barrier()` ensures all state is cleared before
the scan reads it. On later L-tiles the buffer is **kept**, so the recurrence
continues seamlessly from where the previous tile stopped — this is what makes
sequence tiling correct. Cores with no work (`D_start >= D_end`, possible when
`D_inner < NUM_CORES`) still hit the barrier and return
([82-85](SelectiveScan.c#L82-L85)).

### 5.3 Per-channel hoisting `(b, d)`
Loop order is `(b, d, t, n)` — [SelectiveScan.c:88-89](SelectiveScan.c#L88-L89).
For each `(b,d)`:
* **`A_local[16]`** ([91-97](SelectiveScan.c#L91-L97)) — the state row `A[d,:]`
  is time-invariant, so it is copied to the stack once and reused for all `L`
  steps (removes reload + pointer aliasing in the hot loop).
* **`h_local[16]`** ([99-103](SelectiveScan.c#L99-L103)) — the persistent state
  row is loaded from `h_buffer` into a stack array. The whole `t`-scan runs on
  `h_local` (no aliasing, register-friendly) and is written back once at the end
  (§5.7).
* **`d_skip_val`** ([105-106](SelectiveScan.c#L105-L106)) — `D_skip[d]` is
  `t`-invariant and is hoisted out of the `t`-loop, saving one L1 load per step.

### 5.4 Time step — gather selective inputs
For each `t`, [SelectiveScan.c:110-115](SelectiveScan.c#L110-L115) gathers the
inputs of time step `t`: `B_row = B[b,t,:]`, `C_row = C[b,t,:]` (state-indexed,
shared across `d`), and the scalars `x_val = x[b,t,d]`, `z_val = z[b,t,d]`,
`dt_val = dt[b,t,d]`. `y_acc` (the `int64` read-out accumulator $\sum_n C h$) is
reset to 0.

### 5.5 The state recurrence — inner `n`-loop (the heart of the scan)
The lanes are driven by an **unroll scaffold**
([117-123](SelectiveScan.c#L117-L123)) parameterised by `SSM_N_UNROLL`
([19](SelectiveScan.c#L19)); with the current value `1` it is equivalent to a
plain `for (n = 0; n < N; n++)` — the scaffold is retained as a hook for unroll
experiments. Each lane calls `_ssm_h_update`
([120](SelectiveScan.c#L120)) then `_ssm_y_contrib`
([121](SelectiveScan.c#L121)).

Inside `_ssm_h_update`, the LUT-independent $\bar B$ path is computed **first**
so its multiplies overlap the latency of the LUT load on the in-order pipeline:

| Line | Code | SSM meaning |
|------|------|-------------|
| [39](SelectiveScan.c#L39) | `dB_q15 = (dt_val * B_n) >> 8` | $\bar B_t = \Delta_t B_{t,n}$ (simplified Euler) |
| [40](SelectiveScan.c#L40) | `dB_x_q15 = dB_q15 * x_val` | $\bar B_t\, x_t$ (input drive; computed early to hide LUT latency) |
| [41](SelectiveScan.c#L41) | `dt_A_q15 = (dt_val * A_n) >> 8` | $\Delta_t \cdot A_{d,n}$ then $\gg 8$ → $\Delta A$ in Q15 (Q8.8·Q15) |
| [42](SelectiveScan.c#L42) | clip to `[-RANGE, 0]` | keep exp arg in $[-20,0]$; $\Delta A>0 \Rightarrow 0$ (so $\bar A=1$) |
| [43](SelectiveScan.c#L43) | `exp_idx = (clip + RANGE + HALF) >> 8` | round-to-nearest LUT index |
| [44](SelectiveScan.c#L44) | `dA_q15 = SelectiveScan_exp_lut_qwide[exp_idx]` | $\bar A_t = \exp(\Delta A)$ in Q15 (**direct** LUT read) |
| [45](SelectiveScan.c#L45) | `h_acc = (dA_q15 * h_n) >> 15` | $\bar A_t \odot h_{t-1}$ (de-scale Q15 gain) |
| [46](SelectiveScan.c#L46) | `h_acc += dB_x_q15` | $+\,\bar B_t x_t$ → completes $h_t$ |
| [47](SelectiveScan.c#L47) | `return _ssm_sat_i32_i64(h_acc)` | saturate state to int32 |
| [51](SelectiveScan.c#L51) | `(h_new * C_n) >> 15` | $+\,C_{t,n}\,h_t$ (read-out term, in `_ssm_y_contrib`) |

The caller commits the returned $h_t$ into `h_local[n + u]`
([120](SelectiveScan.c#L120)) **before** it feeds `y_acc`
([121](SelectiveScan.c#L121)), so the read-out uses the *post-update* state $h_t$
(Mamba's convention $y_t = C_t h_t$), not $h_{t-1}$.

### 5.6 Skip term, gating, and requantization
After the `n`-loop, still inside the `t`-loop:

1. **Skip / residual $D x$** — [125](SelectiveScan.c#L125):
   `y_acc += d_skip_val * x_val`, completing $y_t = \sum_n C h_t + D x_t$.
2. **SiLU gate** — [126-127](SelectiveScan.c#L126-L127):
   `gate_q13 = gate_lut[z_val + 128]` reads $\operatorname{SiLU}(z_t)$ in Q13;
   `y_gated = round_shift(y_acc * gate_q13, 13)` applies
   $y_t \cdot \operatorname{SiLU}(z_t)$ and strips the Q13 scale. The `z+128`
   offset maps the signed `int8` gate value into the `[0,255]` table.
3. **Output requant** — [128](SelectiveScan.c#L128):
   `y_out = round_shift(y_gated * output_requant_mul_q40, 40)` rescales the wide
   accumulator to the int8 output scale (Q40 multiply + rounding `>>40`).
4. **Saturate & store** — [129-133](SelectiveScan.c#L129-L133): clamp to
   `[-128, 127]` and write `y[b,t,d]`.

### 5.7 Persist state for the next L-tile
[SelectiveScan.c:136-138](SelectiveScan.c#L136-L138). After the full `t`-scan for
`(b,d)`, `h_local` is written back to `h_buffer` so a subsequent L-tile (or the
next call) resumes the recurrence from $h_{L-1}$ (see §5.2).

### 5.8 Final synchronization
[SelectiveScan.c:143](SelectiveScan.c#L143). A closing `pi_cl_team_barrier()`
joins all cores — both the mandatory Deeploy inter-kernel sync point and a
guarantee that every core's `y` / `h_buffer` writes are committed.

---

## 6. Consolidated mapping: SOTA ⇄ kernel

| SSM / Mamba S6 | Discretized form | Kernel expression | Location |
|----------------|------------------|-------------------|----------|
| $\Delta_t$ (selective step) | softplus, Q8.8 | `dt_val` | [114](SelectiveScan.c#L114) |
| $A_{d,n}$ (diag, <0) | static, Q15 | `A_local[n]` → `A_n` | [43](SelectiveScan.c#L43) |
| $\bar A_t = \exp(\Delta_t A)$ | ZOH, Q15 LUT | `dA_q15` | [43-44](SelectiveScan.c#L43-L44) |
| $\bar B_t = \Delta_t B_t$ | simplified Euler | `dB_q15` | [39](SelectiveScan.c#L39) |
| $x_t$ | input | `x_val` | [112](SelectiveScan.c#L112) |
| $h_t = \bar A_t h_{t-1} + \bar B_t x_t$ | recurrence | `h_acc → h_local[n+u]` | [45-47](SelectiveScan.c#L45-L47), [120](SelectiveScan.c#L120) |
| $C_t$ | selective read-out | `C_n` | [51](SelectiveScan.c#L51) |
| $\sum_n C_t h_t$ | read-out | `y_acc` | [121](SelectiveScan.c#L121) |
| $+\,D_d x_t$ | skip | `d_skip_val*x_val` | [125](SelectiveScan.c#L125) |
| $\operatorname{SiLU}(z_t)$ | gate | `gate_lut[z_val+128]` | [126](SelectiveScan.c#L126) |
| $y_t \cdot \operatorname{SiLU}(z_t)$ | gating | `y_gated` | [127](SelectiveScan.c#L127) |
| output requant → int8 | Q40 rescale | `y_out` | [128-133](SelectiveScan.c#L128-L133) |

---

