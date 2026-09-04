"""Does torch distributed rendezvous work across Modal containers, and with which settings?

The 16-rank attempt died with "client socket has timed out ... 1/2 clients joined" connecting to the
rank-0 IPv6 address. Rather than gamble a 14-hour training run on a guess, test rendezvous alone:
16 ranks, one all_reduce, exit. Two candidate fixes are applied together because each is cheap and
independently plausible:
  * USE_LIBUV=0 -- torch >=2.4 defaults TCPStore to a libuv backend whose IPv6 handling is the usual
    culprit on IPv6-only inter-container fabrics.
  * NCCL/GLOO socket family pinned to IPv6, since Modal's i6pn network has no IPv4 route.
Timeout is 180s, not the 900s default, so a failure costs ~3 minutes of GPU instead of 15.
"""
import os
import subprocess

import modal
import modal.experimental

app = modal.App("celeste-modlens-rdzv")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("torch==2.8.0")
       .add_local_file(__file__, "/root/rdzv_test.py", copy=True))

WORKER = r'''
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r % torch.cuda.device_count())
t = torch.ones(1, device="cuda") * r
dist.all_reduce(t)
if r == 0:
    print("RDZV_OK world=%d all_reduce_sum=%.0f expected=%.0f"
          % (w, t.item(), w * (w - 1) / 2), flush=True)
dist.destroy_process_group()
'''


@app.function(image=img, gpu="B200:8", timeout=900)
@modal.experimental.clustered(size=2)
def rdzv():
    from modal.experimental import get_cluster_info
    ci = get_cluster_info()
    rank, ips = ci.rank, ci.container_ips
    open("/root/worker.py", "w").write(WORKER)
    env = dict(os.environ,
               USE_LIBUV="0",                      # libuv TCPStore + IPv6 is the usual failure
               NCCL_SOCKET_FAMILY="AF_INET6",
               GLOO_SOCKET_FAMILY="AF_INET6",
               NCCL_DEBUG="WARN",
               TORCH_NCCL_BLOCKING_WAIT="1")
    cmd = ["torchrun", "--nnodes", str(len(ips)), "--node_rank", str(rank),
           "--master_addr", ips[0], "--master_port", "29901",
           "--nproc_per_node", "8",
           "--rdzv-conf", "timeout=180",
           "/root/worker.py"]
    print("[rdzv] node %d/%d master=%s" % (rank, len(ips), ips[0]), flush=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    print((p.stdout or "")[-1500:], flush=True)
    print((p.stderr or "")[-1500:], flush=True)
    print("[rdzv] node %d exit %d" % (rank, p.returncode), flush=True)
    return p.returncode
