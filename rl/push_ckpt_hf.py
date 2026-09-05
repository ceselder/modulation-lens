"""Push a modulation-lens checkpoint from the Modal volume to the HF hub so it is usable outside.

  modal run rl/push_ckpt_hf.py::main --ckpt /vol/ckpts_modlens_v3/final --repo ceselder/modulation-lens-4bullet-rl
"""
import os, modal

app = modal.App("modlens-push-ckpt")
vol = modal.Volume.from_name("celeste-modlens-vol")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub", "safetensors")
         .add_local_file("rl/push_ckpt_hf.py", "/root/p.py"))


@app.function(image=image, volumes={"/vol": vol}, timeout=3600,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def push(ckpt: str, repo: str, private: bool = True, extra: str = ""):
    from huggingface_hub import HfApi
    tok = os.environ["HF_TOKEN"]
    assert tok, "HF_TOKEN empty"
    api = HfApi(token=tok)
    api.create_repo(repo, private=private, exist_ok=True)
    print("[files]", sorted(os.listdir(ckpt)), flush=True)
    api.upload_folder(folder_path=ckpt, repo_id=repo, commit_message=f"modulation lens from {ckpt}")
    # the prompt is REQUIRED to use the adapter: it carries the single injection marker, and the
    # readout is empty if the marker is missing or the chat template is not re-applied.
    for f in ("prompt.txt",):
        p = os.path.join("/vol/av_sft_4b", f)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=repo)
            print("[+]", f, flush=True)
    if extra:
        api.upload_file(path_or_fileobj=extra.encode(), path_in_repo="USAGE.md", repo_id=repo)
    print("pushed ->", f"https://huggingface.co/{repo}", flush=True)
    return repo


@app.local_entrypoint()
def main(ckpt: str = "/vol/ckpts_modlens_v3/final",
         repo: str = "ceselder/modulation-lens-4bullet-rl", private: bool = True, extra: str = ""):
    push.remote(ckpt=ckpt, repo=repo, private=private, extra=extra)
