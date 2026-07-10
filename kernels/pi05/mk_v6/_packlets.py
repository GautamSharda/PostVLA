from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
P2 = "paligemma_with_expert.gemma_expert.model.layers."
sd2 = torch.load("/network_volume/megakernels/pi05/weights/pi05_libero_merged_fp32.pt", map_location="cpu", weights_only=True)
def L2(i, n): return sd2[P2 + str(i) + "." + n].to(device)
def stack2(n): return torch.stack([L2(i, n) for i in range(18)])
wq2, wk2, wv2 = stack2("self_attn.q_proj.weight"), stack2("self_attn.k_proj.weight"), stack2("self_attn.v_proj.weight")
wo2 = stack2("self_attn.o_proj.weight").half().contiguous()
wg2, wu2 = stack2("mlp.gate_proj.weight"), stack2("mlp.up_proj.weight")
wdn2 = stack2("mlp.down_proj.weight").half().contiguous()
pidx = torch.arange(1024, device=device); hh2, jj2 = pidx // 128, pidx % 128
qsrc2 = torch.stack([hh2 * 256 + jj2, hh2 * 256 + jj2 + 128], 1).reshape(-1)
jk2 = torch.arange(128, device=device); ksrc2 = torch.stack([jk2, jk2 + 128], 1).reshape(-1)
w1_ = torch.cat([wq2[:, qsrc2], wk2[:, ksrc2], wv2], dim=1).half().contiguous()
w2_ = torch.stack([wg2, wu2], dim=2).reshape(18, 8192, 1024).half().contiguous()
def quant2(w):
    s = (w.float().abs().amax(dim=-1).clamp(min=1e-8) / 448.0)
    q = (w.float() / s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).contiguous(), s.float().contiguous()
w1q, s1 = quant2(w1_); woq, so_ = quant2(wo2); w2q, s2 = quant2(w2_); w3q, s3 = quant2(wdn2)
tt2 = torch.tensor([1.0 - i * 0.1 for i in range(10)], dtype=torch.float32, device=device)
temb2 = create_sinusoidal_pos_embedding(tt2, 1024, min_period=4e-3, max_period=4.0, device=device).float()
def lin2(x, wn, bn):
    return torch.nn.functional.linear(x, sd2[wn].float().to(device), sd2[bn].float().to(device))
xa = torch.nn.functional.silu(lin2(temb2, "time_mlp_in.weight", "time_mlp_in.bias"))
cond2 = torch.nn.functional.silu(lin2(xa, "time_mlp_out.weight", "time_mlp_out.bias"))
modsq = torch.zeros(10, 18, 2, 3072, dtype=torch.float32, device=device)
for i in range(18):
    modsq[:, i, 0] = lin2(cond2, P2 + f"{i}.input_layernorm.dense.weight", P2 + f"{i}.input_layernorm.dense.bias")
    modsq[:, i, 1] = lin2(cond2, P2 + f"{i}.post_attention_layernorm.dense.weight", P2 + f"{i}.post_attention_layernorm.dense.bias")
modsq = modsq.contiguous()
modfq = lin2(cond2, "paligemma_with_expert.gemma_expert.model.norm.dense.weight",
             "paligemma_with_expert.gemma_expert.model.norm.dense.bias").contiguous()
w_ain2 = sd2["action_in_proj.weight"].float().to(device).contiguous()
b_ain2 = sd2["action_in_proj.bias"].float().to(device).contiguous()
w_aout2 = sd2["action_out_proj.weight"].float().to(device).contiguous()
b_aout2 = sd2["action_out_proj.bias"].float().to(device).contiguous()
del sd2
