"""Interference functional + the machinery to test whether it *predicts*
cross-domain ablation damage (Workstream 2, docs/INTERFERENCE.md).

The research question
--------------------
`docs/ORTHOGONALITY.md` established that domain routing signatures overlap
more than chance, and that the overlap is structured (python/rust/sql/
math_proof cluster at cosine 0.74-0.90; history sits at -0.25 vs python).
`docs/ABLATION.md` established, on ONE domain pair, that ablating a domain's
experts hurts that domain's text more than a control domain's.

Neither shows that the overlap NUMBER predicts the SIZE of the cross-domain
damage. That quantitative step is what this module exists to test:

    given a routing-overlap score computed BEFORE any ablation,
    how well does it predict how much ablating domain a's experts
    degrades the model on domain b's held-out text?

Relation to the two papers this is positioned against
-----------------------------------------------------
**arXiv 2406.16437, "Theory on Mixture-of-Experts in Continual Learning"
(Li, Lin, Duan, Liang, Shroff; ICLR 2025 Spotlight).** Proves, for
overparameterised *linear regression* tasks, that an MoE diversifies its
experts, that the router learns to select the right expert per task and
balance load, and gives explicit expressions for expected forgetting and
overall generalisation error. Their own abstract states the empirical
extension to DNNs is "experiments on both synthetic and real datasets ...
to extend these insights from linear models to deep neural networks". It is
a theory paper: the forgetting expression is never evaluated against the
measured routing distributions of a real pretrained open-weight LLM.

**arXiv 2503.05029, "Continual Pre-training of MoEs: How robust is your
router?" (Thérien et al.).** Real MoEs (500M-active / 2B-total, 600B tokens).
Finds routing decisions change most in early layers, and that "more
pronounced changes correlate with higher forgetting". That is a *correlation
between an observed post-hoc routing change and forgetting*. It does not fit
a quantitative relationship from a pre-computed routing-overlap score to the
magnitude of cross-task damage, and does not do targeted ablation.

**What this module adds.** The predictive step neither took: compute an
interference number from routing statistics alone, before touching the
model, then measure cross-domain damage causally by ablation, then fit and
permutation-test the relationship. If the number does not predict, the
coefficient and its CI are reported and the claim is not made.

The interference functional
---------------------------
Let E be the set of (layer, expert) slots (1024 for OLMoE-1B-7B-0924), d a
domain, and c_d(e) the number of times expert e was selected on tokens of
domain d, counted under the equal-token-budget control in
`expertatlas.aggregate` (so long-prompt domains do not get extra weight).

    q_d(e) = (c_d(e) + a) / (sum_e' c_d(e') + a|E|)        routing dist. for d
    p(e)   = (sum_d c_d(e) + a) / (grand total + a|E|)     base rate
    l_d(e) = log2( q_d(e) / p(e) )                          == stats.compute_lift

with Laplace a = 1, identical to `expertatlas.stats.compute_lift`.

**Symmetric functional (the primary predictor):**

    I(a, b) = <l_a, l_b> / (||l_a|| * ||l_b||)

This is exactly the quantity in `docs/ORTHOGONALITY.md`; `load_overlap`
below recomputes it with the same code path and *asserts* it reproduces the
published numbers, rather than re-deriving a second, possibly divergent one.

**Directional functional (matched to what the ablation actually does):**

    M(a -> b) = sum_{e in S_a} q_b(e)  /  sum_{e in S_a} p(e)

where S_a is the expert set actually ablated for domain a. M = 1 means
domain b routes through a's ablated experts at exactly the base rate; M > 1
means b leans on a's experts more than an average slice of the network.
I(a,b) is symmetric, damage is not, so this is the asymmetry-aware variant.

Mapping the linear-model math onto empirical routing: what is faithful,
what is our choice
-------------------------------------------------------------------------
2406.16437 works in a setting where each task carries a fixed feature/
representation direction and the model update for a task lives in the
subspace that task's data spans. Forgetting of an earlier task after
training a later one is driven by an inner product between the two tasks'
directions, gated by which experts the router sends each task to: tasks
routed to disjoint experts cannot interfere, and tasks sharing an expert
interfere in proportion to how aligned they are inside it.

FAITHFUL to that structure:
  1. Interference is a **bilinear form (inner product) between two
     per-task vectors**, not a distance or a set-overlap count. Both I and
     M above are inner products.
  2. The vectors are **indexed by expert**, and the router's per-task
     selection probability is what weights each expert's contribution. An
     expert that a domain never routes to contributes zero to that
     domain's interference with anything.
  3. **Zero overlap implies zero predicted interference.** Both functionals
     have that fixed point (I = 0 for orthogonal lift vectors; M = 0 if b
     never routes into S_a).

ADAPTATIONS WE CHOSE, which are judgement calls and not derived from
their theorem:
  A. **The task "representation" is replaced by the routing distribution.**
     Their inner product is between task feature directions inside a shared
     parameter space; we only observe the router, not the per-expert
     representations. So we substitute the *routing* profile for the
     *representation*. This is a strictly weaker object. It is exactly the
     limit `docs/ORTHOGONALITY.md` already flags: two domains routing
     identically could still be orthogonal INSIDE the experts. Our
     predictor cannot see that, and if it under-predicts damage that is one
     candidate reason.
  B. **Base-rate correction (using lift, not raw q).** In their setting
     there is no load-balancing objective forcing a shared, near-uniform
     marginal. In a real trained MoE there is: the marginal p(e) is pushed
     toward uniform by construction, so raw q_a and q_b share a large
     domain-independent component and their raw cosine is near 1 for every
     pair (measured, and reported in docs/INTERFERENCE.md). We therefore
     take the domain-specific *deviation* from the shared component as the
     analogue of the task-specific direction. This is the same reasoning
     the whole project uses for lift, and it is a modelling decision, not a
     theorem. `interference_raw_cosine` computes the uncorrected version so
     the reader can see how much this choice matters.
  C. **log-ratio rather than ratio deviation.** l = log2(q/p) rather than
     (q - p)/p or (q/p - 1). Chosen for consistency with the rest of the
     project (every other number in this repo is in log2-lift units), not
     because their derivation implies a log. `interference_ratio_cosine`
     gives the non-log variant as a robustness check.
  D. **Ablation replaces gradient interference.** Their forgetting is
     caused by a weight update. We do not train; we zero-ablate an expert
     set and measure held-out cross-entropy. This measures *how much of
     domain b's computation flows through a's experts*, which is the
     upper-bound / worst-case version of "how much could learning a
     disturb b". It is a different, harsher operation than a gradient step
     and the magnitudes are not comparable to a forgetting curve.
  E. **One trained model, no task sequence.** They analyse a CL process
     over time; we analyse a static pretrained model's routing geometry.
     Nothing here observes actual forgetting dynamics.

Statistics
----------
The K domains give K*(K-1) ordered (ablator, victim) pairs, but those pairs
are NOT independent observations: they are entries of one KxK matrix, and
every pair shares a domain with 2(K-2) others. A permutation test that
shuffles pair-level y against pair-level x ignores that and is
anticonservative. `mantel_test` permutes **domain labels**, which permutes
rows and columns of the damage matrix jointly and preserves the dependence
structure. For K = 6 it enumerates all 720 relabellings exactly, so the
smallest attainable p-value is 1/720 = 0.0014 -- a hard floor that must be
reported with any p from this design.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

LAPLACE = 1.0


# ---------------------------------------------------------------------------
# Routing profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingProfile:
    """Empirical per-domain routing distributions from a count matrix.

    counts is (n_experts, n_domains) exactly as `aggregate.CountMatrix.counts`.
    """

    domains: list[str]
    expert_uids: list[str]
    q: np.ndarray  # (n_domains, n_experts) P(expert | domain), Laplace-smoothed
    p: np.ndarray  # (n_experts,) P(expert), Laplace-smoothed
    lift: np.ndarray  # (n_experts, n_domains) log2(q/p)

    def index(self, domain: str) -> int:
        return self.domains.index(domain)


def routing_profile(
    counts: np.ndarray,
    domains: list[str],
    expert_uids: list[str],
    laplace: float = LAPLACE,
) -> RoutingProfile:
    """Build q, p and lift from a (n_experts, n_domains) count matrix.

    The lift returned here is bit-identical to `stats.compute_lift(counts)` --
    asserted in tests/ws_int. It is recomputed locally only so that q and p,
    which the directional functional needs, come from the same expressions.
    """
    c = np.asarray(counts, dtype=np.float64)
    n_experts, n_domains = c.shape
    domain_totals = c.sum(axis=0)
    expert_totals = c.sum(axis=1)
    grand = c.sum()

    q = ((c + laplace) / (domain_totals[None, :] + laplace * n_experts)).T  # (D, E)
    p = (expert_totals + laplace) / (grand + laplace * n_experts)  # (E,)
    lift = np.log2(q.T / p[:, None])
    return RoutingProfile(list(domains), list(expert_uids), q, p, lift)


# ---------------------------------------------------------------------------
# The interference functionals (all pre-computed: no ablation involved)
# ---------------------------------------------------------------------------


def _cosine_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vn = v / norms
    return vn @ vn.T


def interference_lift_cosine(profile: RoutingProfile) -> np.ndarray:
    """I(a,b) = cos(l_a, l_b). The primary predictor.

    Identical by construction to `run_orthogonality_analysis.pairwise_cosine`
    applied to `stats.compute_lift` output -- `load_overlap` checks that
    against the published table in docs/ORTHOGONALITY.md.
    """
    return _cosine_rows(profile.lift.T)


def interference_raw_cosine(profile: RoutingProfile) -> np.ndarray:
    """Uncorrected cosine between raw routing distributions q_a, q_b.

    Reported as a diagnostic, not as the predictor: load balancing makes the
    marginal near-uniform, so this is expected to sit near 1.0 for every pair
    and carry almost no discriminative information. Showing that is the point
    -- it is the empirical justification for adaptation (B) in the module
    docstring.
    """
    return _cosine_rows(profile.q)


def interference_ratio_cosine(profile: RoutingProfile) -> np.ndarray:
    """Robustness variant of adaptation (C): cosine of (q_d/p - 1) instead of
    log2(q_d/p). Same fixed point (0 when a domain routes exactly at base
    rate), no log."""
    dev = profile.q / profile.p[None, :] - 1.0
    return _cosine_rows(dev)


def routing_mass_ratio(
    profile: RoutingProfile, expert_set_idx: set[int], victim: str
) -> float:
    """M(a -> b): base-rate-normalised routing mass domain `victim` sends
    through the ablated expert set S_a.

        M = sum_{e in S_a} q_b(e) / sum_{e in S_a} p(e)

    1.0 = victim uses those experts exactly at base rate. Directional, so it
    can distinguish "ablating a hurts b" from "ablating b hurts a", which the
    symmetric cosine cannot.
    """
    if not expert_set_idx:
        return float("nan")
    idx = np.fromiter(sorted(expert_set_idx), dtype=int)
    b = profile.index(victim)
    num = float(profile.q[b, idx].sum())
    den = float(profile.p[idx].sum())
    return num / den if den > 0 else float("nan")


def load_removed(load_ratio: np.ndarray, expert_set_idx: set[int]) -> float:
    """Total utilisation removed by an ablation set, in units of 'fair shares'.

    `load_ratio[e]` (from data/utilization.json, Workstream 3) is expert e's
    share of all selections divided by its fair share 1/|E|. Summing it over
    the ablated set gives how much of the model's actual routed traffic the
    ablation deletes. A size-matched random set has expectation |S| by
    construction, so values below |S| mean the set is colder than average --
    which WS-3 found is true of H1 specialists (enrichment 0.62x into the hot
    set). Carried as a regression covariate so that "overlap predicts damage"
    cannot be an artefact of "this set simply removed more of the network".
    """
    if not expert_set_idx:
        return 0.0
    idx = np.fromiter(sorted(expert_set_idx), dtype=int)
    return float(np.asarray(load_ratio, dtype=np.float64)[idx].sum())


# ---------------------------------------------------------------------------
# Expert-set selection
# ---------------------------------------------------------------------------


def top_m_expert_set(
    lift: np.ndarray,
    significant: np.ndarray,
    domain_index: int,
    m: int,
    min_lift: float = 0.0,
) -> set[int]:
    """The `m` highest-lift experts for a domain among those that are
    BH-FDR-significant and have POSITIVE lift.

    Two deliberate departures from `docs/ABLATION.md`'s rule, both of which
    change the sets and are stated rather than hidden:

    1. **Fixed `m`** rather than "every expert passing the bar" (which gave
       189 vs 170 for its two domains). Fixed size makes every ablation
       remove the same number of experts, so one random-null distribution per
       victim domain is valid for all ablators and no part of a cross-domain
       damage difference can come from having cut more of the network.

    2. **`min_lift` defaults to 0.0, not 1.0.** The project's usual >=2x
       fold-change bar simply cannot fund a fixed set here: on split=A traces
       only 12 experts clear |lift| >= 1.0 for `python` (86 for `history`).
       Requiring it would either force m = 12 or force per-domain sizes back.
       So the bar is relaxed to "FDR-significant and above base rate", and
       the caller is expected to REPORT the lift at rank m per domain so the
       reader can see how strong the marginal member of each set is.
       `docs/INTERFERENCE.md` carries that table.

    Also note `docs/ABLATION.md` selected on `abs(lift) >= 1.0`, which admits
    experts a domain strongly AVOIDS into "that domain's expert set". This
    uses positive lift only. Raises if a domain cannot supply `m`.
    """
    col = np.asarray(lift)[:, domain_index]
    ok = np.asarray(significant)[:, domain_index] & (col >= min_lift) & (col > 0)
    cand = np.flatnonzero(ok)
    if cand.size < m:
        raise ValueError(
            f"domain index {domain_index} has only {cand.size} experts passing "
            f"(FDR-significant AND lift >= {min_lift}); asked for {m}"
        )
    order = cand[np.argsort(-col[cand])]
    return set(int(i) for i in order[:m])


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Null positioning
# ---------------------------------------------------------------------------


def null_position(observed: float, null_samples: np.ndarray) -> dict:
    """Where an observed damage value falls in the matched-size random null.

    Reports the full percentile ladder, not just the mean. docs/TRANSFER.md
    §11's standing warning ("always check percentiles, never just the mean")
    applies in both directions: a null whose mean sits below the observation
    but whose upper tail covers it is not evidence.
    """
    s = np.asarray(null_samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return {"n_null": 0}
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    return {
        "n_null": int(n),
        "null_mean": float(s.mean()),
        "null_std": sd,
        "null_min": float(s.min()),
        "null_p05": float(np.percentile(s, 5)),
        "null_p25": float(np.percentile(s, 25)),
        "null_median": float(np.median(s)),
        "null_p75": float(np.percentile(s, 75)),
        "null_p95": float(np.percentile(s, 95)),
        "null_max": float(s.max()),
        "z": (observed - float(s.mean())) / sd if sd > 0 else float("nan"),
        # fraction of null draws at least as extreme (one-sided, upper)
        "percentile": float((s < observed).mean() * 100.0),
        "empirical_p_upper": float((np.sum(s >= observed) + 1) / (n + 1)),
        "resolution": float(1.0 / (n + 1)),
    }


# ---------------------------------------------------------------------------
# Fitting the predictive relationship
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z CI for a Pearson r.

    Reported with an explicit warning wherever it is used: it assumes n
    independent observations. Ordered domain pairs from a KxK matrix are not
    independent, so this interval is TOO NARROW. `domain_jackknife` and
    `mantel_test` are the honest ones.
    """
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    crit = float(norm.ppf(1 - alpha / 2))
    lo, hi = z - crit * se, z + crit * se
    return (math.tanh(lo), math.tanh(hi))


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope/intercept plus Pearson r, Spearman rho, R^2."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3 or np.std(x) == 0:
        return {"n": int(n), "slope": float("nan"), "intercept": float("nan"),
                "pearson_r": float("nan"), "spearman_rho": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    r = _pearson(x, y)
    rho = float(spearmanr(x, y).statistic)
    return {
        "n": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": float(r),
        "spearman_rho": rho,
        "r2": float(r * r),
        "fisher_ci_naive": fisher_ci(r, n),
        "y_std": float(np.std(y, ddof=1)),
        "x_std": float(np.std(x, ddof=1)),
    }


def bootstrap_ci_pairs(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile bootstrap over PAIRS. Same independence problem as the
    Fisher interval -- reported for comparison only, labelled as such."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    rs, slopes = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0:
            continue
        rs.append(_pearson(xb, yb))
        slopes.append(np.polyfit(xb, yb, 1)[0])
    rs = np.asarray([v for v in rs if np.isfinite(v)])
    slopes = np.asarray([v for v in slopes if np.isfinite(v)])
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {
        "r_ci": (float(np.percentile(rs, q[0])), float(np.percentile(rs, q[1]))) if rs.size else (float("nan"),) * 2,
        "slope_ci": (float(np.percentile(slopes, q[0])), float(np.percentile(slopes, q[1]))) if slopes.size else (float("nan"),) * 2,
        "n_boot": int(rs.size),
    }


def domain_jackknife(
    domains: list[str], x_pairs: dict, y_pairs: dict
) -> list[dict]:
    """Leave-one-DOMAIN-out recomputation of the correlation.

    With K domains the regression has K(K-1) points but only K independent
    units. If dropping a single domain moves r a lot, the relationship is
    that domain, not a law. This is the single most informative honesty check
    available at this sample size, and it is cheap.
    """
    out = []
    for drop in domains:
        keep = [d for d in domains if d != drop]
        xs = [x_pairs[k] for k in x_pairs if k[0] in keep and k[1] in keep]
        ys = [y_pairs[k] for k in x_pairs if k[0] in keep and k[1] in keep]
        fit = fit_linear(np.asarray(xs), np.asarray(ys))
        out.append({"dropped": drop, **{k: fit[k] for k in ("n", "pearson_r", "spearman_rho", "slope")}})
    return out


def mantel_test(
    domains: list[str],
    x_pairs: dict,
    y_pairs: dict,
    n_perm: int = 0,
    seed: int = 0,
) -> dict:
    """Permutation test that respects the matrix dependence structure.

    Null: the labelling of domains in the damage matrix is exchangeable with
    respect to the overlap matrix. Each permutation applies one relabelling
    pi of the K domains to BOTH indices of the damage entries, i.e.
    y'(a,b) = y(pi(a), pi(b)), and recomputes the correlation against the
    unchanged x. This is the standard Mantel construction; it holds the
    within-matrix dependence fixed, unlike shuffling the K(K-1) pair values
    independently.

    n_perm = 0 enumerates all K! relabellings exactly (720 for K = 6). The
    two-sided p-value therefore cannot go below 1/K!, which is 0.0014 at
    K = 6 -- the design's power floor, and it is reported, not hidden.
    """
    keys = [k for k in x_pairs if k in y_pairs]
    x = np.asarray([x_pairs[k] for k in keys], dtype=np.float64)
    y = np.asarray([y_pairs[k] for k in keys], dtype=np.float64)
    obs = _pearson(x, y)

    if not np.isfinite(obs):
        return {"observed_r": float("nan"), "p_two_sided": float("nan")}

    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        perms = (rng.permutation(len(domains)) for _ in range(n_perm))
        exact = False
        total = n_perm
    else:
        perms = (np.asarray(p) for p in itertools.permutations(range(len(domains))))
        exact = True
        total = math.factorial(len(domains))

    d_index = {d: i for i, d in enumerate(domains)}
    ge = 0
    n_used = 0
    null_rs = []
    for perm in perms:
        relabel = {domains[i]: domains[perm[i]] for i in range(len(domains))}
        yp = []
        ok = True
        for (a, b) in keys:
            k2 = (relabel[a], relabel[b])
            if k2 not in y_pairs:
                ok = False
                break
            yp.append(y_pairs[k2])
        if not ok:
            continue
        r = _pearson(x, np.asarray(yp, dtype=np.float64))
        if not np.isfinite(r):
            continue
        null_rs.append(r)
        n_used += 1
        if abs(r) >= abs(obs) - 1e-12:
            ge += 1

    null_rs = np.asarray(null_rs)
    return {
        "observed_r": float(obs),
        "n_permutations_used": int(n_used),
        "exact_enumeration": bool(exact),
        "p_two_sided": float(ge / n_used) if n_used else float("nan"),
        "p_floor": float(1.0 / total),
        "null_r_mean": float(null_rs.mean()) if null_rs.size else float("nan"),
        "null_r_std": float(null_rs.std(ddof=1)) if null_rs.size > 1 else float("nan"),
        "null_r_p95_abs": float(np.percentile(np.abs(null_rs), 95)) if null_rs.size else float("nan"),
    }


def partial_correlation(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Correlation of x with y after regressing the covariate z out of both.

    Used for Workstream 3's confound: if cross-domain damage is really driven
    by how much *load* an ablation removed rather than by routing overlap,
    the partial correlation of damage with overlap given load-removed
    collapses toward 0 while the raw one does not.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if np.std(z) == 0:
        return _pearson(x, y)
    A = np.column_stack([np.ones_like(z), z])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return _pearson(rx, ry)


def multiple_regression(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """OLS with t-stats. Standard errors assume independent rows, which
    ordered domain pairs are not -- flagged wherever reported."""
    from scipy.stats import t as t_dist

    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    n, k = A.shape
    dof = n - k
    if dof <= 0:
        return {"names": ["intercept"] + names, "beta": beta.tolist()}
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * (1 - t_dist.cdf(np.abs(tvals), dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "names": ["intercept"] + names,
        "beta": [float(v) for v in beta],
        "se_naive": [float(v) for v in se],
        "t": [float(v) for v in tvals],
        "p_naive": [float(v) for v in pvals],
        "r2": float(r2),
        "n": int(n),
        "dof": int(dof),
    }


# ---------------------------------------------------------------------------
# Matched-LOAD nulls (Workstream 2, load/overlap disentanglement)
# ---------------------------------------------------------------------------


def matched_load_null_sets(
    load_ratio: np.ndarray,
    target_set: set[int],
    n_sets: int,
    rng: np.random.Generator,
    tolerance: float = 0.02,
    max_tries_per_set: int = 4000,
    eligible: set[int] | None = None,
) -> tuple[list[set[int]], dict]:
    """Random expert sets matched to `target_set` on BOTH size and total load.

    Why this exists
    ---------------
    Every null in this project so far is **size**-matched: draw |S| experts
    uniformly. But Workstream 3 measured that H1 specialists are
    disproportionately *cold* (enrichment 0.62x into the hot set), and the
    per-domain load actually removed spans **5.3x** across the six domains
    (sql 192.4 fair-shares vs history 36.3, against a random expectation of
    100). A size-matched null therefore removes ~100 fair-shares by
    construction while `ablate_sql` removes 192 -- so "sql's ablation hurt
    more" is confounded with "sql's ablation deleted twice as much of the
    routed network".

    A size-matched null cannot separate those. This one can: it holds total
    load fixed and lets only *which* experts differ, so an effect surviving
    it is about expert identity rather than quantity of network removed.

    Method
    ------
    Swap-based rejection sampling. Start from a uniform draw, then repeatedly
    swap one member for a non-member that moves total load toward the target.
    Accept when within `tolerance` (relative). This is a matched-sampling
    design, not a reweighting: every returned set is a genuine expert set of
    exactly |target_set| members.

    Returns
    -------
    (sets, diagnostics). **Always inspect diagnostics.** If
    `achieved_relative_error` is large or `n_returned < n_sets`, the target
    load is not reachable at this set size (e.g. a very hot target set may
    exceed what any 100 experts can sum to) and the null is not valid -- report
    that rather than using it.
    """
    load = np.asarray(load_ratio, dtype=np.float64)
    n_experts = load.size
    universe = np.arange(n_experts) if eligible is None else np.fromiter(sorted(eligible), dtype=int)
    size = len(target_set)

    # Validate BEFORE touching load[], or an out-of-range member raises
    # IndexError from load_removed instead of this clearer error.
    if size == 0 or size > universe.size:
        raise ValueError(f"cannot draw {size} experts from {universe.size} eligible")
    if target_set and (min(target_set) < 0 or max(target_set) >= n_experts):
        raise ValueError(
            f"target_set contains an expert index outside [0, {n_experts})"
        )

    target_load = load_removed(load, target_set)

    # Feasibility: the coldest and hottest achievable sums at this size.
    srt = np.sort(load[universe])
    lo, hi = float(srt[:size].sum()), float(srt[-size:].sum())
    feasible = lo <= target_load <= hi

    sets: list[set[int]] = []
    errors: list[float] = []
    tries_used: list[int] = []

    for _ in range(n_sets):
        cur = set(rng.choice(universe, size=size, replace=False).tolist())
        cur_load = load_removed(load, cur)
        tries = 0
        while abs(cur_load - target_load) > tolerance * max(target_load, 1e-9):
            if tries >= max_tries_per_set:
                break
            members = np.fromiter(cur, dtype=int)
            outside = np.setdiff1d(universe, members, assume_unique=False)
            if outside.size == 0:
                break
            need_more = cur_load < target_load
            drop = members[np.argmax(load[members])] if not need_more else members[np.argmin(load[members])]
            # Pick a replacement that moves total load the right way.
            cand = outside[load[outside] > load[drop]] if need_more else outside[load[outside] < load[drop]]
            if cand.size == 0:
                break
            add = int(rng.choice(cand))
            cur.discard(int(drop))
            cur.add(add)
            cur_load = load_removed(load, cur)
            tries += 1

        rel = abs(cur_load - target_load) / max(target_load, 1e-9)
        if rel <= tolerance:
            sets.append(cur)
            errors.append(rel)
            tries_used.append(tries)

    diagnostics = {
        "target_load": float(target_load),
        "set_size": int(size),
        "feasible": bool(feasible),
        "feasible_range": [lo, hi],
        "n_requested": int(n_sets),
        "n_returned": len(sets),
        "achieved_relative_error_mean": float(np.mean(errors)) if errors else float("nan"),
        "achieved_relative_error_max": float(np.max(errors)) if errors else float("nan"),
        "swaps_mean": float(np.mean(tries_used)) if tries_used else float("nan"),
        "tolerance": tolerance,
    }
    if not feasible:
        diagnostics["warning"] = (
            f"target load {target_load:.1f} outside achievable range "
            f"[{lo:.1f}, {hi:.1f}] at size {size} — matched-load null is INVALID"
        )
    elif len(sets) < n_sets:
        diagnostics["warning"] = (
            f"only {len(sets)}/{n_sets} sets converged within tolerance — "
            "treat the null as underpowered"
        )
    return sets, diagnostics
