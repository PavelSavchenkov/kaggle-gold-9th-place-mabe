import hashlib
import json

import numpy as np
import pandas as pd
import torch  # type: ignore
from tqdm import tqdm

from dl.configs import FeaturesConfigDL
from dl.data_balancer import DataBalancerForLabeledTrain
from dl.features_lookup_gpu import FeaturesLookupGPU
from dl.landmark_feature_extractor import LandmarkFeatureExtractor

CNT_SAMPLES_FOR_STATS = 1_000_000


def calc_hash_for_stats_from_feats_dict(
    meta: pd.DataFrame,
    features_config: dict,
) -> str:
    video_ids = sorted(int(v) for v in set(meta["video_id"].tolist()))
    payload = {
        "video_ids": video_ids,
        "features_config": features_config,
        "target_samples": CNT_SAMPLES_FOR_STATS,
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.md5(payload_json.encode("utf-8")).hexdigest()  # length 32


def calc_hash_for_stats(
    meta: pd.DataFrame,
    features_config: FeaturesConfigDL,
) -> str:
    cfg_dict = features_config.model_dump(mode="json")
    return calc_hash_for_stats_from_feats_dict(meta, features_config=cfg_dict)


def calc_stats(
    balancer: DataBalancerForLabeledTrain,
    lookup: FeaturesLookupGPU,
) -> dict[str, np.ndarray]:
    features_config = lookup.feats_config
    landmark_extractor = LandmarkFeatureExtractor(
        features_config, aug_config=lookup.aug_config
    )

    device = torch.device("cuda")

    # ---------- Streaming mean / std over numeric_feats ----------
    B = 1024
    running_sum = None  # [D]
    running_sum_sq = None  # [D]
    total_positions = 0  # scalar, counts B * T

    with torch.no_grad():
        for offset in tqdm(
            range(0, CNT_SAMPLES_FOR_STATS, B), desc="Pre-Calculating mean/std stats"
        ):
            curr_bs = min(B, CNT_SAMPLES_FOR_STATS - offset)

            video_ids_batch: list[int] = []
            agents_batch: list[int] = []
            targets_batch: list[int] = []
            frames_batch: list[int] = []

            for j in range(curr_bs):
                s = balancer.get_sample(epoch=0, idx=offset + j)
                video_ids_batch.append(int(s.video_id))
                agents_batch.append(int(s.agent))
                targets_batch.append(int(s.target))
                frames_batch.append(int(s.frame))

            batch = {
                "video_id": torch.tensor(
                    video_ids_batch, dtype=torch.long, device=device
                ),
                "agent": torch.tensor(agents_batch, dtype=torch.long, device=device),
                "target": torch.tensor(targets_batch, dtype=torch.long, device=device),
                "frame": torch.tensor(frames_batch, dtype=torch.long, device=device),
            }

            batch = lookup.preprocess_batch(batch, is_train=False)
            batch = landmark_extractor(batch)
            numeric_feats: torch.Tensor = batch["numeric_feats"]  # [B, T, D_num]

            B, T, D = numeric_feats.shape
            flat = numeric_feats.reshape(B * T, D).to(torch.float64)

            if running_sum is None:
                running_sum = flat.sum(dim=0)  # [D]
                running_sum_sq = (flat * flat).sum(dim=0)  # [D]
            else:
                running_sum += flat.sum(dim=0)
                running_sum_sq += (flat * flat).sum(dim=0)

            total_positions += flat.size(0)

    assert running_sum is not None and running_sum_sq is not None
    assert total_positions > 0

    mean = (running_sum / total_positions).cpu().numpy()  # [D]
    mean_sq = (running_sum_sq / total_positions).cpu().numpy()
    var = np.maximum(mean_sq - mean**2, 1e-12)
    std = np.sqrt(var)

    stats = {
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
    }
    return stats
