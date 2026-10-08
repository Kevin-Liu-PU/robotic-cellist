"""alpha / actor std / sampled-vs-deterministic gap per checkpoint, evaluated on the run's own replay obs."""
import sys, glob, pickle, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import SAC
run = sys.argv[1]
buf = pickle.load(open(glob.glob(f"{run}/*_latest_buffer.pkl")[0], "rb"))
n = buf.pos if not buf.full else buf.buffer_size
obs_all = buf.observations[:n].reshape(n, -1)
print(f"{run.split('/')[-1]}  buffer n={n}")
print("ckpt      alpha    std    |a_det|  |a_s-a_d|  H_gauss")
for z in sorted(glob.glob(f"{run}/ckpt_ep*.zip")) + glob.glob(f"{run}/*_latest.zip"):
    m = SAC.load(z, device="cpu")
    k = min(n, int(m.num_timesteps))
    obs = torch.as_tensor(obs_all[:k], dtype=torch.float32)
    with torch.no_grad():
        mean, log_std, _ = m.policy.actor.get_action_dist_params(obs)
        std = log_std.exp(); a_det = torch.tanh(mean)
        a_s = torch.tanh(mean + std * torch.randn_like(std))
        H = (0.5 * np.log(2 * np.pi * np.e) + log_std).sum(1).mean()
    alpha = float(m.log_ent_coef.exp()) if getattr(m, "log_ent_coef", None) is not None else float(m.ent_coef)
    name = z.split('/')[-1].replace('.zip', '').replace('sac_piece_twinkle-short_', '')
    print(f"{name:9s} {alpha:.4f}  {float(std.mean()):.3f}  {float(a_det.abs().mean()):.3f}    {float((a_s-a_det).abs().mean()):.3f}     {float(H):+.2f}")
