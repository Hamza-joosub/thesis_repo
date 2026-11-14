# Cross-validation of FP primitives against Matthews (2024)

**Purpose.** Verify the flexible-probability routines in `thesis_model.py` against the independent open-source MATLAB implementation accompanying Matthews (2024), *Hierarchical Portfolio Stress-Testing with Flexible Probabilities*, MSc dissertation, University of Cape Town (supervisor: T. Gebbie). Repository: `github.com/nematthews/Meucci-HS-FP-framework-SA-context`.

**Method.** Matthews's `es_probs.m` (exponential time-decay probabilities) and `ens.m` (effective number of scenarios, standard and generalised forms) were executed **unmodified** under GNU Octave 8.4 on toy inputs, with outputs written to CSV and compared against the corresponding constructions in `thesis_model.py` (Stage 1 weights: `exp(λt)/Σ`, λ = ln2/τ; ENS: `exp(−Σ pₜ ln pₜ)`). One environment shim was required and is disclosed: Octave does not implement MATLAB's `height()`; a one-line shim (`height(x) = size(x,1)`) was placed on the path. No function under test was edited.

**Parameterisation note.** The two implementations parameterise identically up to normalisation: Matthews weights by `exp(−(ln2/τ)(T−t))`, this study by `exp(λt)` with λ = ln2/τ; the normalised probability vectors coincide.

## Results

| Test | Max \|difference\| | Verdict |
|---|---|---|
| Weight vector, T=100, τ=10 | 2.8 × 10⁻¹⁷ | PASS |
| Weight vector, T=1,060, τ=118 (thesis sample & half-life) | 6.1 × 10⁻¹⁸ | PASS |
| ENS, T=100 (= 38.9214497604) | 1.4 × 10⁻¹⁴ | PASS |
| ENS, T=1,060 (= **456.1818197701**) | 8.0 × 10⁻¹³ | PASS |
| Generalised ENS (γ=2), T=100 | 1.1 × 10⁻¹⁴ | PASS |
| Generalised ENS (γ=2), T=1,060 | 5.1 × 10⁻¹³ | PASS |
| ENS, hand-made ragged vector | 0 | PASS |
| Generalised ENS (γ=2), ragged vector | 0 | PASS |

All eight comparisons agree to well within 10⁻¹² (double-precision noise). The thesis's reported effective sample size (ENS ≈ 456 at T = 1,060 weeks, half-life 118 weeks) is reproduced exactly by the independent implementation.

**Incidental finding.** `es_probs.m` defaults to τ = 24 months — the reference implementation's own default half-life is two years, consistent with the half-life adopted in this study.

**Scope.** This validates the time-conditioning and ENS layers, which are the layers in active use. Matthews's kernel (`kl_probs.m`), crisp (`cr_probs.m`) and entropy-pooling (`ep_probs.m`) routines remain available as oracles for the state-conditioning layer if it is reinstated.

**Thesis citation sentence (Section 3, Stage 5 / diagnostics):**
> The flexible-probability and effective-sample-size routines were cross-validated against the independent open-source implementation accompanying Matthews (2024), agreeing to machine precision on all test cases, including at this study's sample size and half-life.

**Reproduction.** `validate.m` (Octave harness), `height.m` (shim), and the comparison script are included alongside this file; run order: `octave validate.m` then `python compare.py`.

---

# Part 2 — Real-data validation (thesis_data.csv, 1,310 weekly obs, Mar 2001–Apr 2026)

Run 1 Sep 2026 against `thesis_model_reviewed.py` (standalone copy `tm.py`; app imports stripped, `CSV_PATH` local). Baseline specification: ALSI log-return on {MSCI Materials, MSCI EM, MSCI Staples, ZARUSD} log-returns; τ = 118 weeks; estimation from index 250 (first date 2005-12-23).

| # | Test | Oracle | Result |
|---|---|---|---|
| 1a | Regression layer, uniform weights, α→0 | `statsmodels` OLS | coefficients, intercept, R² agree to 3×10⁻¹³ |
| 1b | Regression layer, exponential weights, α→0 | `statsmodels` WLS (same weights) | coefficients, intercept, weighted R², ENS agree to 4×10⁻¹³ |
| 2 | Closed-form weighted LOO R² | brute-force leave-one-out, weights & standardisation held fixed | **exact** (diff 0.00) |
| 2′ | Approximation scope | brute-force LOO with weights re-normalised and standardisation recomputed per left-out obs | 4.6×10⁻⁶ — disclosed, negligible |
| 3 | Raw-vs-standardised coefficient back-transformation | prediction identity | 1.4×10⁻¹⁷ |
| 4 | Lookahead | corrupt all rows after the estimation date, refit | fit unchanged — no leak; slice inclusive of estimation date (contemporaneous by design) |
| 5 | ENS path, **all 1,060 estimation dates** | Matthews (2024) `es_probs.m` + `ens.m`, verbatim, Octave | max diff 3.2×10⁻¹² |
| 6a | OLS baseline (`lamb=None`) | runs (previously crashed) | ✔ |
| 6b | Per-channel stability diagnostics | dict populated for all four channels | ✔ |
| 6c | Caller list not mutated by interaction terms | length preserved | ✔ |
| 7 | Per-date α selection (`rolling_fit_ridge_FP_auto_alpha`) | end-to-end, 1,060 dates, 11 s | ✔ |
| 8 | Drop-one ΔR² (`drop_one_delta_r2`) | end-to-end at final date | ✔ |

## Findings surfaced by the validation run

- **ENS path:** 230 (Dec 2005, history-limited) → 461 (Apr 2026); reproduced by the independent implementation at every date.
- **Penalty selection:** LOO-optimal α ∈ {1, 3.16, 10, 31.6} across the sample (median 10), never at the grid boundary and never below 1 — out-of-sample fit prefers shrinkage to OLS throughout.
- **Alternative-estimator comparison (for §4.4):** mean LOO R² — ridge (auto-α) 0.668; ridge (fixed α=0.5) 0.668; OLS 0.645. Ridge improves out-of-sample explanatory power by ~2.3 points; results insensitive to α over 0.5–30.
- **Overfit flags:** 0.0% under time-weighting alone — the earlier 2.49% rate was a Stage 2 (state-conditioning) artefact.
- **Drop-one ΔR² at 2026-04-10 (α=31.6, full LOO R² 0.669):** Materials +0.103, EM +0.039, Staples +0.003, **ZARUSD +0.0007** — the currency channel's incremental contribution has effectively vanished by sample end, corroborating the rand-hedge collapse seen in the beta path via an independent metric. Recommended §4.2 exhibit: the ΔR² path over the full sample.
- **Metric caution:** CV is uninformative for channels whose beta path crosses zero by design (Staples: CV 2.0, sign-flip 0.5%); report sign-flip rate and the path.
