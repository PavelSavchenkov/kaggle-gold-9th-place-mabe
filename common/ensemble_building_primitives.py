import copy
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import torch  # type: ignore
from cir_model import CenteredIsotonicRegression  # type: ignore
from tqdm import tqdm

from common.helpers import ensure_1d_numpy, logit_to_prob, prob_to_logit


class EnsembleObjective(str, Enum):
    nested_f1 = "nested_f1"
    f1 = "f1"
    pr_auc = "pr_auc"
    logloss = "logloss"


@dataclass
class Ensemble:
    model_idx: list[int]
    weights: list[float]
    score: float  # bigger is better
    bias: float | None = (
        None  # if not None, then it's log regression and .infer should act accordingly
    )
    logit_space: bool = False

    def infer(self, preds_list: list[np.ndarray]) -> np.ndarray:
        assert preds_list

        s = np.zeros_like(preds_list[0])
        for w, idx in zip(self.weights, self.model_idx):
            pred = (
                preds_list[idx]
                if not self.logit_space
                else prob_to_logit(preds_list[idx])
            )
            s += pred * w

        if self.bias is not None:
            # logreg
            s = s + self.bias
            return 1.0 / (1.0 + np.exp(-s))
        else:
            if self.logit_space:
                s = logit_to_prob(s)
            return s

    def size(self) -> int:
        return len(self.model_idx)

    def are_model_types_different(self, model_types: list[str]) -> bool:
        types = [model_types[i] for i in self.model_idx]
        return len(set(types)) == len(types)

    def extend(self, idx: int, w: float, score: float) -> "Ensemble":
        new_e = copy.deepcopy(self)
        new_e.weights = [old_w * (1.0 - w) for old_w in self.weights]
        if idx in new_e.model_idx:
            new_e.weights[new_e.model_idx.index(idx)] += w
        else:
            new_e.model_idx.append(idx)
            new_e.weights.append(w)
        new_e.score = score
        return new_e

    def is_near_duplicate(self, other: "Ensemble", tol: float = 1e-2) -> bool:
        if set(self.model_idx) != set(other.model_idx):
            return False
        for i in range(len(self.model_idx)):
            j = other.model_idx.index(self.model_idx[i])
            if abs(self.weights[i] - other.weights[j]) > tol:
                return False
        return True


def get_top_k_ensembles(
    es: list[Ensemble],
    k: int,
) -> list[Ensemble]:
    """
    Select top-k ensembles by score, filtering only near-duplicates (same model set + similar weights).
    """
    assert k >= 0
    es = list(sorted(es, key=lambda e: -e.score))
    top_k_es: list[Ensemble] = []

    for e in es:
        if len(top_k_es) == k:
            break
        skip = False
        for taken_e in top_k_es:
            if taken_e.is_near_duplicate(e):
                skip = True
                break
        if skip:
            continue
        top_k_es.append(e)

    return top_k_es


def _create_weights_range(
    num_bins: int, allow_neg_weights: bool = False
) -> torch.Tensor:
    assert num_bins >= 2
    if not allow_neg_weights:
        w = torch.linspace(0.0, 1.0, steps=num_bins, device="cuda", dtype=torch.float32)
    else:
        w = torch.linspace(
            -1.0, 1.0, steps=num_bins, device="cuda", dtype=torch.float32
        )
    w = w[torch.abs(w) > 1e-3]
    return w


@torch.no_grad()
def _calc_objective(
    preds: torch.Tensor,
    labels: torch.Tensor,
    objective: EnsembleObjective,
    fold_id: torch.Tensor | None = None,
    inputs_are_logits: bool = False,
) -> torch.Tensor:
    """
    preds: [N] or [K, N] float32
    labels: [N] int (0/1)
    returns: [K] scores (float32)
    """
    if preds.ndim == 1:
        preds = preds.unsqueeze(0)  # [1, N]
    K, N = preds.shape

    if inputs_are_logits:
        preds = logit_to_prob(preds)

    if labels.dtype != torch.int:
        labels = labels.int()
    labels_bin = (labels > 0).int()
    total_pos_global = labels_bin.sum().clamp_min(1)

    if objective == EnsembleObjective.logloss:
        # Binary log-loss (cross-entropy) averaged over N per model.
        # Assumes preds are probabilities in [0, 1].
        eps = 1e-7
        preds_clamped = preds.clamp(min=eps, max=1.0 - eps)

        # Broadcast labels to [K, N]
        labels_f = labels_bin.view(1, N).expand_as(preds_clamped).to(torch.float32)

        # per-example loss: - [ y*log(p) + (1-y)*log(1-p) ]
        loss = -(
            labels_f * preds_clamped.log()
            + (1.0 - labels_f) * (1.0 - preds_clamped).log()
        )
        # mean over N, result [K]
        logloss = loss.mean(dim=1)
        return -logloss.to(torch.float32)

    _, idx_sorted = preds.sort(dim=1, descending=True)  # [K, N]

    labels_expanded = labels_bin.view(1, N).expand(K, N)
    labels_sorted = torch.gather(labels_expanded, 1, idx_sorted)  # [K, N]
    labels_sorted_f = labels_sorted.to(torch.float32)

    tp_cum_global = labels_sorted_f.cumsum(dim=1)  # [K, N]
    ranks_global = torch.arange(1, N + 1, device="cuda", dtype=torch.float32).view(1, N)

    if objective == EnsembleObjective.f1:
        denom = ranks_global + total_pos_global
        f1 = 2.0 * tp_cum_global / denom
        best, _ = f1.max(dim=1)
        return best.to(torch.float32)

    elif objective == EnsembleObjective.pr_auc:
        precision = tp_cum_global / ranks_global
        ap = (precision * labels_sorted_f).sum(dim=1) / total_pos_global
        return ap.to(torch.float32)
    elif objective == EnsembleObjective.nested_f1:
        if fold_id is None:
            raise ValueError("fold_id is required for nested_f1")

        # Align fold_ids to global prediction rank
        # This allows us to mask the sorted array directly
        fold_expanded = fold_id.view(1, N).expand(K, N)
        fold_sorted = torch.gather(fold_expanded, 1, idx_sorted)  # [K, N]

        unique_folds = fold_id.unique()
        assert len(unique_folds) > 1
        scores_sum = torch.zeros(K, device="cuda", dtype=torch.float32)

        for test_fold in unique_folds:
            test_mask = fold_sorted == test_fold  # [K, N] boolean mask

            # 2. Calculate Test Stats (Cumulative)
            # We need these to later subtract them from global stats
            # tp_cum_test[k, i] = num positives in fold `fid` among top `i` global predictions
            tp_cum_test = (labels_sorted_f * test_mask.float()).cumsum(dim=1)
            # ranks_test[k, i] = num samples in fold `fid` among top `i` global predictions
            ranks_test = test_mask.float().cumsum(dim=1)

            # 3. Derive Train Stats (The Subtraction Trick)
            # Train = Global - Test
            tp_cum_train = tp_cum_global - tp_cum_test
            ranks_train = ranks_global - ranks_test

            # Calculate total positives specifically for the train set
            total_pos_test_scalar = (labels_bin * (fold_id == test_fold)).sum()
            total_pos_train = total_pos_global - total_pos_test_scalar

            # 4. Optimize Threshold on Train
            # F1 = 2TP / (Preds + TotalPos)
            denom_train = ranks_train + total_pos_train.clamp_min(1)
            f1_train = 2.0 * tp_cum_train / denom_train

            # Find the global rank index `i` that maximizes F1 on the training set
            _, best_idx = f1_train.max(dim=1, keepdim=True)  # [K, 1]

            # 5. Evaluate on Test Fold using that threshold
            # We gather the test stats at the specific optimal index found on train
            tp_test_final = torch.gather(tp_cum_test, 1, best_idx).squeeze(1)
            preds_test_final = torch.gather(ranks_test, 1, best_idx).squeeze(1)

            denom_test = preds_test_final + total_pos_test_scalar

            # Calculate final F1 for this fold
            # clamp denominator to avoid div by zero if fold is empty
            fold_f1 = 2.0 * tp_test_final / denom_test.clamp_min(1e-6)

            scores_sum += fold_f1

        return scores_sum / len(unique_folds)
    else:
        raise ValueError(f"Unknown objective: {objective}")


def _calc_weighted_sum_ensemble(
    scores_stack: torch.Tensor, e: Ensemble
) -> torch.Tensor:
    """
    scores_stack: [M, N] float32 cuda. M = #models, N = #samples
    returns: [N] ensemble predictions
    """
    idx = torch.as_tensor(e.model_idx, device="cuda", dtype=torch.long)
    w = torch.as_tensor(e.weights, device="cuda", dtype=torch.float32)
    sel = scores_stack.index_select(0, idx)  # [K, N]
    return torch.einsum("k,kn->n", w, sel)  # [N]


@torch.no_grad()
def _get_ensemble_extensions(
    scores_stack: torch.Tensor,
    e_init: Ensemble,
    weights_range: torch.Tensor,
    labels: torch.Tensor,
    objective: EnsembleObjective,
    fold_id: torch.Tensor | None = None,
    inputs_are_logits: bool = False,
) -> list[Ensemble]:
    """
    Given a base ensemble `e_init`, try all (model_idx, weight) pairs
    and return one best extension per model. Skip if no better than e_init
    """
    M, N = scores_stack.shape
    W = len(weights_range)
    total = M * W

    free_bytes, _ = torch.cuda.mem_get_info("cuda:0")
    mem_cap = free_bytes
    # print(f"mem_cap = {mem_cap/2**30:.2f}GB")

    approx_mem_per_iteration = N * 4
    if objective == "nested_f1":
        approx_mem_per_iteration *= 25
    else:
        approx_mem_per_iteration *= 15
    batch_size = max(1, int(mem_cap / approx_mem_per_iteration))
    # print(f"N={N}, batch_size={batch_size}")

    base_scores = _calc_weighted_sum_ensemble(scores_stack, e_init)  # [N]

    scores_all = torch.empty(total, device="cuda", dtype=torch.float32)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        k = end - start

        idx_flat = torch.arange(start, end, device="cuda")
        m_idx = torch.div(idx_flat, W, rounding_mode="floor")
        w_idx = torch.remainder(idx_flat, W)

        scores_chunk = scores_stack.index_select(0, m_idx)  # [k, N]
        w_chunk = weights_range.index_select(0, w_idx).view(k, 1)  # [k, 1]

        combos_chunk = (
            base_scores.unsqueeze(0) * (1.0 - w_chunk) + scores_chunk * w_chunk
        )  # [k, N]

        scores_chunk = _calc_objective(
            combos_chunk,
            labels,
            fold_id=fold_id,
            objective=objective,
            inputs_are_logits=inputs_are_logits,
        )  # [k]
        scores_all[start:end] = scores_chunk

    es: list[Ensemble] = []
    for m in range(M):
        scores_chunk = scores_all[m * W : (m + 1) * W]
        best_w_idx = int(torch.argmax(scores_chunk).item())
        best_w = float(weights_range[best_w_idx].item())
        best_score = float(scores_chunk[best_w_idx].item())

        e_ext = e_init.extend(idx=m, w=best_w, score=best_score)

        if e_ext.score <= e_init.score + 1e-6 and e_ext.size() == e_init.size():
            continue

        es.append(e_ext)

    return es


@torch.no_grad()
def build_ensembles_beam_search(
    y_preds: list[np.ndarray],
    y_true: np.ndarray,
    num_bins: int,
    pool_size: int,
    max_models: int,
    objective: EnsembleObjective,
    fold_id: np.ndarray | None = None,
    allow_neg_weights: bool = False,
    model_types: (
        list[str] | None
    ) = None,  # if not None, consider only ensemble from models of different types
    verbose: bool = True,
    use_logits: bool = False,
) -> dict[int, Ensemble]:
    M = len(y_preds)
    assert M > 0

    N = len(y_preds[0])

    if use_logits:
        logits_y_preds = []
        for y_pred in y_preds:
            logits_y_preds.append(prob_to_logit(y_pred))
        y_preds = logits_y_preds

    scores_stack = torch.stack(
        [
            torch.from_numpy(np.asarray(s, dtype=np.float32).reshape(-1)).to("cuda")
            for s in y_preds
        ],
        dim=0,
    )

    labels = torch.from_numpy(y_true.astype(np.int32)).to("cuda")
    assert labels.shape[0] == N

    weights_range = _create_weights_range(num_bins, allow_neg_weights)

    if fold_id is not None:
        fold_id = torch.from_numpy(fold_id.astype(np.int8)).to("cuda")

    best_e: dict[int, Ensemble] = {}
    pools: dict[int, list[Ensemble]] = defaultdict(list)

    # init with single-model ensembles
    for m in range(M):
        base_scores = scores_stack[m]  # [N]
        objective_score = _calc_objective(
            base_scores,
            labels,
            fold_id=fold_id,
            objective=objective,
            inputs_are_logits=use_logits,
        )
        e = Ensemble(
            model_idx=[m],
            weights=[1.0],
            score=float(objective_score.item()),
            logit_space=use_logits,
        )
        pools[1].append(e)

    cnt_models = 1
    while cnt_models <= max_models and cnt_models <= M:
        if cnt_models not in pools or not pools[cnt_models]:
            cnt_models += 1
            continue

        if verbose:
            print(f"Iterate pool for cnt_models={cnt_models}")

        for e in pools[cnt_models]:
            size = e.size()
            if size not in best_e or best_e[size].score < e.score:
                best_e[size] = copy.deepcopy(e)

        if verbose:
            print(
                f"So far best for cnt_models={cnt_models}: "
                f"{best_e[cnt_models].score:.5f}"
            )

        next_es: list[Ensemble] = []
        for e in tqdm(
            get_top_k_ensembles(
                pools[cnt_models],
                k=pool_size,
            ),
            desc=f"Iterate pool for cnt_models={cnt_models}",
            disable=not verbose,
        ):
            next_e_candidates = _get_ensemble_extensions(
                scores_stack,
                e,
                weights_range,
                labels,
                fold_id=fold_id,
                objective=objective,
                inputs_are_logits=use_logits,
            )
            for e in next_e_candidates:
                if model_types is None or e.are_model_types_different(model_types):
                    next_es.append(e)

        pools[cnt_models].clear()
        for e in next_es:
            pools[e.size()].append(e)

        cnt_models += 1

    for k in range(2, max_models + 1):
        if k not in best_e or best_e[k - 1].score > best_e[k].score:
            best_e[k] = copy.deepcopy(best_e[k - 1])

    return best_e


# ----- Experimental -----


def _fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "sklearn",  # "cuml" or "sklearn"
):
    """
    Fit binary logistic regression and return:
      w: (D,) coefficients
      b: scalar intercept
      score: -logloss on (X, y)
    """
    y = np.asarray(y).astype(np.float32).ravel()
    X = np.asarray(X, dtype=np.float32)
    assert X.shape[0] == y.shape[0]

    if backend == "cuml":
        # GPU via RAPIDS cuML
        from cuml.linear_model import LogisticRegression as cuLogReg  # type: ignore

        clf = cuLogReg(
            penalty="l2",
            C=1.0,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-4,
            verbose=0,
        )
        clf.fit(X, y)

        w = np.asarray(clf.coef_).ravel()
        b = float(np.asarray(clf.intercept_).ravel()[0])

    else:
        # CPU via scikit-learn
        from sklearn.linear_model import LogisticRegression as SkLogReg

        clf = SkLogReg(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            fit_intercept=True,
        )
        clf.fit(X, y)

        w = clf.coef_.ravel()
        b = float(clf.intercept_[0])

    # Compute -logloss as score (higher = better)
    logits = X @ w + b
    proba = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-15
    proba = np.clip(proba, eps, 1.0 - eps)
    logloss = -np.mean(y * np.log(proba) + (1.0 - y) * np.log(1.0 - proba))
    score = -float(logloss)

    return w, b, score


def build_ensembles_logreg(
    y_preds: list[np.ndarray],
    y_true: np.ndarray,
    max_models: int,
    verbose: bool = True,
    backend: str = "sklearn",
) -> dict[int, Ensemble]:
    assert len(y_preds) > 0, "y_preds must be non-empty"

    # X_full: (N, M)
    preds_flat = [np.asarray(p, dtype=np.float32).ravel() for p in y_preds]
    n_models = len(preds_flat)
    X_full = np.stack(preds_flat, axis=1)
    y = np.asarray(y_true).ravel()
    assert X_full.shape[0] == y.shape[0], "All preds must have same length as y_true"

    max_k = min(max_models, n_models)

    # 1) Fit full logistic regression
    w_full, b_full, score_full = _fit_logreg(X_full, y, backend=backend)

    if verbose:
        print(f"[full] k={n_models}, score=-logloss={score_full:.6f}")

    # 2) Order models by |beta|
    order = np.argsort(-np.abs(w_full))  # descending |w|

    ensembles: dict[int, Ensemble] = {}

    # 3) For each k, refit on top-k models
    for k in range(1, max_k + 1):
        selected = order[:k]
        X_k = X_full[:, selected]

        w_k, b_k, score_k = _fit_logreg(X_k, y, backend=backend)

        ensemble = Ensemble(
            model_idx=selected.tolist(),
            weights=w_k.tolist(),
            score=score_k,
            bias=b_k,
        )
        ensembles[k] = ensemble

        if verbose:
            print(
                f"[top-k] k={k:2d}, score=-logloss={score_k:.6f}, "
                f"models={ensemble.model_idx}"
            )

    return ensembles


class ProbCalibrator:
    model: CenteredIsotonicRegression

    def __init__(self):
        pass

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred = ensure_1d_numpy(y_pred)
        y_true = ensure_1d_numpy(y_true)
        y_true = y_true.astype(y_pred.dtype, copy=False)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape} != {y_true.shape}"
        assert y_pred.dtype == y_true.dtype, f"{y_pred.dtype} != {y_true.dtype}"
        self.model = CenteredIsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self.model.fit(y_pred, y_true)

    def fit_transform(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_pred=y_pred, y_true=y_true)
        return self.transform(y_pred=y_pred)

    @staticmethod
    def from_path(path: Path) -> "ProbCalibrator":  # type: ignore
        obj = ProbCalibrator()
        obj.load(path)
        return obj

    def load(self, path: Path):
        self.model = joblib.load(path)

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        orig_shape = y_pred.shape
        y_pred = ensure_1d_numpy(y_pred)
        res = self.model.transform(y_pred)
        res = res.reshape(orig_shape)
        return res

    def save(self, dir: Path, name: str) -> Path:
        dst = dir / f"{name}.joblib"
        joblib.dump(self.model, dst)
        return dst
