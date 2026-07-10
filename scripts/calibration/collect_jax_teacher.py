#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.so100_mujoco import So100MujocoEnv
from rlinf.models.embodiment.jax_openpi import JaxOpenPiPolicy

POSTVLA_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get('POSTVLA_ARTIFACT_ROOT', POSTVLA_ROOT / 'artifacts'))
OUT = Path(os.environ.get('POSTVLA_TEACHER_DATASET', ARTIFACT_ROOT / 'teacher_dataset'))
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = os.environ.get(
    'POSTVLA_CUBE_MANIFEST',
    str(POSTVLA_ROOT / 'configs' / 'manifests' / 'cubevar_contact_40.json'),
)
TEACHER_CKPT = os.environ.get(
    'POSTVLA_JAX_TEACHER_CKPT',
    str(ARTIFACT_ROOT / 'checkpoints' / 'openpi_jax_sft' / '7999'),
)
SIM_ROOT = os.environ.get('POSTVLA_SIM_ROOT', str(POSTVLA_ROOT / 'sim'))
EPISODES = int(os.environ.get('POSTVLA_TEACHER_EPISODES', '20'))
PROGRESS = OUT / 'progress.json'

print('out', OUT, flush=True)
print('teacher', TEACHER_CKPT, flush=True)

model_cfg = OmegaConf.create({
    'model_path': TEACHER_CKPT,
    'action_dim': 6,
    'num_action_chunks': 16,
    'openpi': {'config_name': 'pi05_so100_sim'},
})
env_cfg = OmegaConf.create({
    'seed': 0,
    'num_envs': 1,
    'flashact_demo_root': SIM_ROOT,
    'cube_xy_manifest': MANIFEST,
    'control_fps': 30,
    'steps_per_control': None,
    'max_episode_steps': 900,
    'ignore_terminations': True,
    'reset_warmup_frames': 0,
    'prompt': 'pick up the blue cube and place it on the pink pad',
    'on_pad_half_extent': 0.139,
    'pad_center_threshold': 0.03,
    'lift_z_threshold': 0.13,
    'cube_rest_z_threshold': 0.065,
})

def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def scalar_bool(x, default=False):
    if x is None:
        return default
    return bool(to_np(x).reshape(-1)[0])

def scalar_float(x, default=float('nan')):
    if x is None:
        return default
    arr = to_np(x).reshape(-1)
    return float(arr[0]) if arr.size else default

print('loading teacher...', flush=True)
teacher = JaxOpenPiPolicy(model_cfg)
print('teacher loaded', flush=True)
env = So100MujocoEnv(env_cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
print('env ready positions', len(env._cube_xy_positions), flush=True)

results = []
start = time.time()
try:
    for episode_id in range(EPISODES):
        ep_path = OUT / f'episode_{episode_id:03d}.npz'
        if ep_path.exists():
            print('skip existing', episode_id, flush=True)
            continue
        t0 = time.time()
        obs, info = env.reset(seed=episode_id, options={'episode_id': episode_id})
        initial_cube = env._cube_pos(env.datas[0])[:3].copy()
        for _ in range(12):
            obs, _, _, _, info = env.step(None, auto_reset=False)
        main_images=[]; wrist_images=[]; states=[]; actions=[]; steps=[]
        step = 0
        while step < 900:
            main_images.append(to_np(obs['main_images'])[0].astype(np.uint8))
            wrist_images.append(to_np(obs['wrist_images'])[0].astype(np.uint8))
            states.append(to_np(obs['states'])[0].astype(np.float32))
            steps.append(step)
            chunk, _ = teacher.predict_action_batch(obs, mode='eval')
            chunk = np.asarray(chunk, dtype=np.float32)
            actions.append(chunk[0])
            for a in chunk[0].astype(np.float64):
                obs, _, term, trunc, info = env.step(a[None, :], auto_reset=False)
                step += 1
                if step >= 900:
                    break
        ep = info.get('episode', {}) if isinstance(info, dict) else {}
        item = {
            'episode_id': episode_id,
            'initial_cube_xyz': [float(x) for x in initial_cube],
            'samples': len(actions),
            'steps': step,
            'on_pad': scalar_bool(ep.get('on_pad_once', ep.get('success_once'))),
            'pad_center_3cm_lift': scalar_bool(ep.get('pad_center_3cm_lift_once')),
            'max_cube_z': scalar_float(info.get('max_cube_z') if isinstance(info, dict) else None),
            'cube_target_xy_distance': scalar_float(info.get('cube_target_xy_distance') if isinstance(info, dict) else None),
            'file': str(ep_path),
            'wall_s': round(time.time() - t0, 3),
        }
        np.savez_compressed(
            ep_path,
            main_images=np.stack(main_images),
            wrist_images=np.stack(wrist_images),
            states=np.stack(states),
            actions=np.stack(actions),
            steps=np.asarray(steps, dtype=np.int32),
            episode_id=np.asarray([episode_id], dtype=np.int32),
            initial_cube_xyz=np.asarray(initial_cube, dtype=np.float32),
        )
        results.append(item)
        done_files = sorted(OUT.glob('episode_*.npz'))
        progress = {
            'teacher_checkpoint': TEACHER_CKPT,
            'manifest': MANIFEST,
            'output_dir': str(OUT),
            'episodes_done': len(done_files),
            'new_results': results,
            'elapsed_s': round(time.time() - start, 3),
        }
        PROGRESS.write_text(json.dumps(progress, indent=2))
        xyz = ','.join(f'{v:.3f}' for v in item['initial_cube_xyz'])
        print(f"episode {episode_id:02d}: cube=({xyz}) samples={item['samples']} on_pad={item['on_pad']} strict={item['pad_center_3cm_lift']} dist={item['cube_target_xy_distance']:.4f} max_z={item['max_cube_z']:.4f} wall={item['wall_s']}s files={len(done_files)}", flush=True)
finally:
    env.close()
print('done', OUT, flush=True)
