from typing import Dict
import torch
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

if __name__=="__main__":
    zarr_path = 'data/pusht/pusht_cchi_v7_replay.zarr'
    replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
    print(replay_buffer.n_episodes)