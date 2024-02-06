if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    
from typing import Dict
import torch
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

if __name__=="__main__":
#     zarr_path = 'data/pusht/pusht_cchi_v7_replay.zarr'
    zarr_path = 'data/maniskill/blocks_100/zarr'
    replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
    print(replay_buffer.n_episodes)