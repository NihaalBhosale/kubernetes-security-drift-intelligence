"""Learned components of the proposed solution.

DRWA  - Dynamic Rule Weight Adaptation (Algorithm 2, learned part). Interpretable, linear in rule evidence:

            g_hat = sum_t [ w+_t(theta) * log1p(pos_ctx_t)  -  w-_t(theta) * log1p(neg_ctx_t) ]
            w_t(theta) = lambda * w0_t  +  delta_t  +  sum_j kappa_tj * phi_j(theta)

        w0_t are the base paper's Table VI control values (the prior); delta_t and kappa_tj are learned by
        ridge regression that shrinks towards that prior. phi_j are the z-scored environment parameters
        selected by the Parameter Identification Algorithm, so the effective weight of every rule changes with
        the cluster it is evaluated in. Target g = sign(dD) * log1p(|dD| / 0.25) ("log-damage").

HybridModel - gradient boosting on [rule evidence, graph deltas, theta]; the final "proposed" detector.
RuleInducer - decision-tree rule induction (new compound rules from data, human readable).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from app.dynamic.params import PARAM_NAMES
from app.dynamic.templates import TEMPLATES, TEMPLATE_IDS

TAU = 0.25


def log_damage(dd) -> np.ndarray:
    dd = np.asarray(dd, dtype=float)
    return np.sign(dd) * np.log1p(np.abs(dd) / TAU)


def cols(prefix: str) -> List[str]:
    return [f"{prefix}::{t}" for t in TEMPLATE_IDS]


THETA_COLS = [f"theta::{p}" for p in PARAM_NAMES]
GRAPH_COLS = ["d_ext_max", "d_ext_top5", "d_ext_sum", "d_p1_max", "d_p1_top5", "d_p1_sum", "d_mean_reach",
              "d_high_reach", "d_pods", "n_new_total", "n_gone_total", "n_templates_new", "n_templates_gone"]
EVIDENCE_COLS = cols("pos_ctx") + cols("neg_ctx") + cols("pos_raw") + cols("neg_raw")
FULL_COLS = EVIDENCE_COLS + GRAPH_COLS + THETA_COLS
W0 = np.array([t.w0 for t in TEMPLATES])


# ----------------------------------------------------------------------------------- metrics
def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximising F1 on validation data (all methods tuned the same way)."""
    qs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    best, bt = -1.0, float(qs[0])
    for t in qs:
        f = f1_score(y, s >= t, zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def metrics(y: np.ndarray, score: np.ndarray, thr: float, dD: Optional[np.ndarray] = None) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    out = {
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
        "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "alert_rate": float(pred.mean()),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = roc_auc_score(y, score)
        out["pr_auc"] = average_precision_score(y, score)
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    if dD is not None and np.std(score) > 0:
        out["spearman"] = float(spearmanr(score, dD)[0])
    else:
        out["spearman"] = float("nan") if dD is not None else None
    return out


# ------------------------------------------------------------------------------------- DRWA
class DRWA:
    def __init__(self, phi: Sequence[str], alpha: float = 30.0):
        self.phi = list(phi)                      # parameter names (without "theta::")
        self.alpha = alpha
        self.mu: Dict[str, float] = {}
        self.sd: Dict[str, float] = {}
        self.lam = 1.0
        self.ridge: Optional[Ridge] = None
        self.n_t = len(TEMPLATES)

    def _phi(self, df: pd.DataFrame) -> np.ndarray:
        P = np.column_stack([np.clip((df[f"theta::{p}"].values - self.mu[p]) / self.sd[p], -3, 3) for p in self.phi])
        return P

    def _design(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        A = np.log1p(df[cols("pos_ctx")].values)            # n x T
        Bm = np.log1p(df[cols("neg_ctx")].values)
        prior = (A * W0).sum(1) - (Bm * W0).sum(1)
        P = self._phi(df)                                    # n x K
        feats = [A, -Bm]
        for j in range(P.shape[1]):
            feats.append(A * P[:, [j]])
            feats.append(-Bm * P[:, [j]])
        return np.hstack(feats), prior

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "DRWA":
        for p in self.phi:
            col = df[f"theta::{p}"].values
            self.mu[p], self.sd[p] = float(col.mean()), float(col.std() or 1.0)
        X, prior = self._design(df)
        g = log_damage(y)
        self.lam = float(max(0.0, (prior @ g) / max(prior @ prior, 1e-9)))     # scale of the Table VI prior
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, g - self.lam * prior)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, prior = self._design(df)
        return self.lam * prior + self.ridge.predict(X)

    # -- the dynamic weights themselves ---------------------------------------------------------
    def weights(self, theta: Dict[str, float]) -> pd.DataFrame:
        """Effective weight of every rule template for the cluster described by theta."""
        T, K = self.n_t, len(self.phi)
        coef = self.ridge.coef_
        z = np.array([np.clip((theta[p] - self.mu[p]) / self.sd[p], -3, 3) for p in self.phi])
        base_pos, base_neg = coef[:T], -coef[T:2 * T]
        w_pos = self.lam * W0 + base_pos
        w_neg = self.lam * W0 + base_neg
        for j in range(K):
            off = 2 * T + j * 2 * T
            w_pos = w_pos + coef[off:off + T] * z[j]
            w_neg = w_neg - coef[off + T:off + 2 * T] * z[j]
        return pd.DataFrame({"template": TEMPLATE_IDS, "prior_w0": W0, "w_introduced": w_pos, "w_removed": w_neg})


# --------------------------------------------------------------------------------- Hybrid GBM
@dataclass
class HybridModel:
    clf: Optional[HistGradientBoostingClassifier] = None
    reg: Optional[HistGradientBoostingRegressor] = None
    cols: List[str] = field(default_factory=lambda: list(FULL_COLS))
    relevance: Dict[str, float] = field(default_factory=dict)
    seed: int = 0

    def fit(self, df: pd.DataFrame, y: np.ndarray, dD: np.ndarray) -> "HybridModel":
        X = df[self.cols].values
        self.clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6, min_samples_leaf=20,
                                                  l2_regularization=1.0, random_state=self.seed).fit(X, y)
        self.reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=6, min_samples_leaf=20,
                                                 l2_regularization=1.0, random_state=self.seed).fit(X, log_damage(dD))
        return self

    def proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[self.cols].values)[:, 1]

    def damage(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(df[self.cols].values)

    def param_relevance(self, df_val: pd.DataFrame, y_val: np.ndarray, repeats: int = 3) -> Dict[str, float]:
        """Relevance of each environment parameter = AUC lost when it is permuted (validation data)."""
        X = df_val[self.cols].copy()
        idx = [self.cols.index(c) for c in THETA_COLS]
        base = roc_auc_score(y_val, self.clf.predict_proba(X.values)[:, 1])
        rng = np.random.default_rng(self.seed)
        rel = {}
        for c in THETA_COLS:
            drops = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[c] = rng.permutation(Xp[c].values)
                drops.append(base - roc_auc_score(y_val, self.clf.predict_proba(Xp.values)[:, 1]))
            rel[c.split("::", 1)[1]] = float(np.mean(drops))
        self.relevance = rel
        return rel


# -------------------------------------------------------------------------------- rule induction
class RuleInducer:
    """Induce human-readable compound rules (IF .. AND .. THEN dangerous) with a shallow decision tree."""

    def __init__(self, max_depth: int = 6, min_leaf: int = 25):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, class_weight="balanced",
                                           random_state=0)
        self.names: List[str] = []

    def _X(self, df: pd.DataFrame, theta_cols: Sequence[str]) -> np.ndarray:
        ev = (df[cols("pos_ctx")].values > 0).astype(float)                 # rule fired (introduced)
        ng = (df[cols("neg_ctx")].values > 0).astype(float)
        ctx = np.log1p(df[cols("pos_ctx")].values)
        self.names = [f"NEW[{t}]" for t in TEMPLATE_IDS] + [f"REMOVED[{t}]" for t in TEMPLATE_IDS] + \
                     [f"ctx[{t}]" for t in TEMPLATE_IDS] + list(theta_cols)
        return np.hstack([ev, ng, ctx, df[list(theta_cols)].values])

    def fit(self, df: pd.DataFrame, y: np.ndarray, theta_cols: Sequence[str]) -> "RuleInducer":
        self.theta_cols = list(theta_cols)
        self.tree.fit(self._X(df, theta_cols), y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.tree.predict_proba(self._X(df, self.theta_cols))[:, 1]

    def rules(self, df: pd.DataFrame, y: np.ndarray, min_precision: float = 0.7, min_support: int = 15) -> List[dict]:
        X = self._X(df, self.theta_cols)
        t = self.tree.tree_
        leaves = self.tree.apply(X)
        out = []

        def walk(node: int, conds: List[str]):
            if t.children_left[node] == -1:
                m = leaves == node
                sup = int(m.sum())
                if sup >= min_support:
                    prec = float(y[m].mean())
                    if prec >= min_precision:
                        out.append({"if": " AND ".join(conds) or "TRUE", "support": sup, "precision": prec})
                return
            f, thr = t.feature[node], t.threshold[node]
            nm = self.names[f]
            walk(t.children_left[node], conds + [f"{nm} <= {thr:.2f}"])
            walk(t.children_right[node], conds + [f"{nm} > {thr:.2f}"])

        walk(0, [])
        return sorted(out, key=lambda r: -r["support"])


def save(obj, path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
