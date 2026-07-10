#!/usr/bin/env python3
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from omegaconf import OmegaConf
from openpi.models import model as _model
from rlinf.models.embodiment.openpi import get_model

POSTVLA_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get('POSTVLA_ARTIFACT_ROOT', POSTVLA_ROOT / 'artifacts'))
DATA=Path(os.environ.get('POSTVLA_TEACHER_DATASET', ARTIFACT_ROOT / 'teacher_dataset'))
SOURCE=Path(os.environ.get('POSTVLA_TORCH_SFT_CKPT', ARTIFACT_ROOT / 'checkpoints' / 'torch_sft'))
TARGET=Path(os.environ.get('POSTVLA_DISTILLED_CKPT', ARTIFACT_ROOT / 'checkpoints' / 'pi05_so100_sim_distilled_sft'))
OUT=Path(os.environ.get('POSTVLA_CALIBRATION_RUN', ARTIFACT_ROOT / 'calibration_run'))
OUT.mkdir(parents=True, exist_ok=True)
PROMPT='pick up the blue cube and place it on the pink pad'
SUMMARY=OUT/'summary.json'

EPOCHS=int(os.environ.get('POSTVLA_CALIBRATION_EPOCHS', '3'))
BATCH_SIZE=int(os.environ.get('POSTVLA_CALIBRATION_BATCH_SIZE', '4'))
LR_LAYER=float(os.environ.get('POSTVLA_CALIBRATION_LAYER_LR', '1e-5'))
LR_HEAD=float(os.environ.get('POSTVLA_CALIBRATION_HEAD_LR', '5e-5'))
WEIGHT_DECAY=0.0
SEED=17


def cfg_model(path):
    return OmegaConf.create({
        'model_path': str(path),
        'precision': None,
        'model_type': 'openpi',
        'num_action_chunks': 16,
        'action_dim': 6,
        'is_lora': False,
        'lora_rank': 32,
        'use_proprio': True,
        'num_steps': 10,
        'add_value_head': True,
        'openpi': {
            'config_name': 'pi05_so100_sim', 'num_images_in_input': 2, 'noise_level': 0.5,
            'action_chunk': 16, 'num_steps': 10, 'train_expert_only': True,
            'action_env_dim': 6, 'noise_method': 'flow_noise', 'add_value_head': True,
            'value_after_vlm': True, 'value_vlm_mode': 'mean_token', 'detach_critic_input': None,
            'use_dsrl': False, 'dsrl_state_dim': 8, 'dsrl_action_noise_dim': 32,
            'dsrl_num_q_heads': 10, 'dsrl_agg_q': 'mean', 'dsrl_image_latent_dim': 64,
            'dsrl_state_latent_dim': 64, 'dsrl_hidden_dims': [128,128,128],
            'action_horizon': 16, 'noise_params': [0.16,0.12,200], 'joint_logprob': True,
        },
        'policy_setup': 'so100_mujoco',
    })

files=sorted(DATA.glob('episode_*.npz'))
arrays=[]; index=[]; ep_success={}
for fi,f in enumerate(files):
    z=np.load(f)
    arrays.append(z)
    for j in range(z['actions'].shape[0]): index.append((fi,j))
progress_path=DATA/'progress.json'
if progress_path.exists():
    prog=json.load(open(progress_path))
    for r in prog.get('new_results', []): ep_success[int(r['episode_id'])]=bool(r['on_pad'])
print('data_files', len(files), 'samples', len(index), 'teacher_successes', sum(ep_success.values()), flush=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
random.shuffle(index)
val_n=max(64, int(0.10*len(index)))
val=index[:val_n]
train=index[val_n:]
print('train', len(train), 'val', len(val), flush=True)

print('loading model', flush=True)
model=get_model(cfg_model(SOURCE)).cuda().train()
for p in model.parameters(): p.requires_grad_(False)
layer_params=[]; head_params=[]; train_names=[]
for name,p in model.named_parameters():
    is_layer=('gemma_expert.model.layers.16.' in name or 'gemma_expert.model.layers.17.' in name)
    is_head=(name.startswith('action_out_proj') or name.startswith('action_in_proj') or name.startswith('time_mlp_'))
    if is_layer or is_head:
        p.requires_grad_(True)
        train_names.append(name)
        (layer_params if is_layer else head_params).append(p)
print('train_tensors', len(train_names), 'layer_tensors', len(layer_params), 'head_tensors', len(head_params), flush=True)
opt=torch.optim.AdamW([
    {'params': layer_params, 'lr': LR_LAYER},
    {'params': head_params, 'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)


def make_batch(sel):
    main=[]; wrist=[]; states=[]; acts=[]
    for fi,j in sel:
        z=arrays[fi]
        main.append(z['main_images'][j]); wrist.append(z['wrist_images'][j]); states.append(z['states'][j]); acts.append(z['actions'][j])
    obs={'observation/image': np.stack(main), 'observation/wrist_image': np.stack(wrist), 'observation/state': np.stack(states), 'actions': np.stack(acts), 'prompt': [PROMPT]*len(sel)}
    processed=model.input_transform(obs, transpose=False)
    actions=processed.pop('actions').to(device='cuda', dtype=torch.float32)
    processed=model.precision_processor(processed)
    observation=_model.Observation.from_dict(processed)
    return observation, actions

@torch.no_grad()
def eval_loss(sel, max_batches=32):
    model.eval()
    losses=[]
    sub=list(sel[:max_batches*BATCH_SIZE])
    for i in range(0, len(sub), BATCH_SIZE):
        obs, acts=make_batch(sub[i:i+BATCH_SIZE])
        loss=model.sft_forward((obs, acts), use_action_chunk_loss=True)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float('nan')

base_val=eval_loss(val)
print('base_val_loss', base_val, flush=True)
history=[]
start=time.time()
for epoch in range(1, EPOCHS+1):
    random.shuffle(train)
    losses=[]; t_epoch=time.time()
    for step in range(0, len(train), BATCH_SIZE):
        batch=train[step:step+BATCH_SIZE]
        obs, acts=make_batch(batch)
        opt.zero_grad(set_to_none=True)
        loss=model.sft_forward((obs, acts), use_action_chunk_loss=True)
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        nstep=step//BATCH_SIZE + 1
        if nstep % 50 == 0 or step + BATCH_SIZE >= len(train):
            print(f'epoch {epoch} step {nstep}/{(len(train)+BATCH_SIZE-1)//BATCH_SIZE} loss={losses[-1]:.5f} mean={np.mean(losses):.5f} grad={float(gn.detach().cpu() if torch.is_tensor(gn) else gn):.4f} mem_gb={torch.cuda.max_memory_allocated()/1e9:.2f}', flush=True)
    val_loss=eval_loss(val)
    row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':val_loss,'wall_s':round(time.time()-t_epoch,3)}
    history.append(row)
    partial={
        'kind':'jax_teacher_cubevar20_900_tail2_sft_calibration',
        'source':str(SOURCE),'target':str(TARGET),'dataset':str(DATA),
        'epochs':EPOCHS,'batch_size':BATCH_SIZE,'lr_layer':LR_LAYER,'lr_head':LR_HEAD,
        'samples':len(index),'train_samples':len(train),'val_samples':len(val),
        'base_val_loss':base_val,'history':history,'elapsed_s':round(time.time()-start,3),
        'train_tensors':len(train_names),'train_names_sample':train_names[:20],
    }
    SUMMARY.write_text(json.dumps(partial, indent=2))
    print('epoch_done', row, flush=True)

print('saving checkpoint', TARGET, flush=True)
TARGET.mkdir(parents=True, exist_ok=True)
for name in ['config.json']:
    shutil.copy2(SOURCE/name, TARGET/name)
for dname in ['flashact','assets']:
    src=SOURCE/dname
    dst=TARGET/dname
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)
safetensors.torch.save_model(model, TARGET/'model.safetensors')
summary=json.loads(SUMMARY.read_text())
summary['saved']=True
summary['model_safetensors']=str(TARGET/'model.safetensors')
summary['final_val_loss']=history[-1]['val_loss'] if history else None
SUMMARY.write_text(json.dumps(summary, indent=2))
(TARGET/'calibration_summary.json').write_text(json.dumps(summary, indent=2))
print('done', SUMMARY, flush=True)
