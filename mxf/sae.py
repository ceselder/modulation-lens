"""BatchTopK SAE (adamkarvonen/qwen3-8b-saes, dictionary_learning format) — minimal loader.

The conditioning/reward direction for feature f is the UNIT ENCODER COLUMN unit(W_enc[:,f])
(probe side — same space as the cluster-probe directions, drops into inject@1/read@L as-is).
Scoring uses raw per-sample encoder activations, never the batch-topk gate: we want a smooth
reward signal below the firing threshold.
"""
import torch
from huggingface_hub import hf_hub_download

from mxf.config import D_MODEL, READ_LAYER

SAE_REPO = "adamkarvonen/qwen3-8b-saes"
SAE_FILENAME = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{READ_LAYER}/trainer_2/ae.pt"
MAX_ACTS_REPO = "adamkarvonen/sae_max_acts"          # dataset repo
MAX_ACTS_FILENAME = (f"acts_Qwen_Qwen3-8B_layer_{READ_LAYER}_trainer_2_"
                     "layer_percent_75_context_length_32.pt")


class BatchTopKSAE:
    """W_enc [d,F], W_dec [F,d], b_enc [F], b_dec [d]."""

    def __init__(self, W_enc, W_dec, b_enc, b_dec):
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = W_enc, W_dec, b_enc, b_dec
        self.d_in, self.d_sae = W_enc.shape

    def encode_features(self, acts_BLD, feature_ids):
        """Pre-topk post-ReLU encoder activations relu((x-b_dec)@W_enc[:,f]+b_enc[f]).
        acts_BLD [B,L,d] -> [B,L,len(feature_ids)]."""
        idx = torch.as_tensor(feature_ids, device=self.W_enc.device)
        return torch.relu((acts_BLD - self.b_dec) @ self.W_enc[:, idx] + self.b_enc[idx])

    def enc_dirs(self, feature_ids):
        """Unit encoder columns [len(ids), d] — the conditioning/reward directions."""
        idx = torch.as_tensor(feature_ids, device=self.W_enc.device)
        return torch.nn.functional.normalize(self.W_enc[:, idx].T, dim=-1)


def load_sae(path=None, device="cpu", dtype=torch.float32):
    """path=None resolves via hf_hub_download (uses HF_HOME cache on the box)."""
    path = path or hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILENAME)
    params = torch.load(path, map_location="cpu", weights_only=False)
    key_map = {"encoder.weight": "W_enc", "decoder.weight": "W_dec", "encoder.bias": "b_enc",
               "bias": "b_dec", "b_dec": "b_dec"}   # dictionary_learning aliases for b_dec
    t = {key_map[k]: v.to(dtype) for k, v in params.items() if k in key_map}
    sae = BatchTopKSAE(t["W_enc"].T.contiguous().to(device),  # nn.Linear stores [out, in]
                       t["W_dec"].T.contiguous().to(device),
                       t["b_enc"].to(device), t["b_dec"].to(device))
    assert sae.d_in == D_MODEL, f"SAE d_in {sae.d_in} != D_MODEL {D_MODEL}"
    nrm = sae.W_dec.norm(dim=1)
    assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-2), "decoder rows must be unit norm"
    return sae


def load_max_acts(path=None):
    """{"max_tokens": [F,N,L] long, "max_acts": [F,N,L] float} on cpu. path=None resolves via HF."""
    path = path or hf_hub_download(repo_id=MAX_ACTS_REPO, filename=MAX_ACTS_FILENAME,
                                   repo_type="dataset")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert "max_tokens" in data and "max_acts" in data, f"unexpected keys: {list(data.keys())}"
    return data
