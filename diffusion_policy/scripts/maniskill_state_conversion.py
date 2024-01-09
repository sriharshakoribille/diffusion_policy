if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import h5py
import os
import click
import pathlib
from pathlib import Path
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import gzip
import json
from typing import Sequence, Union

def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret

# python3 diffusion_policy/scripts/maniskill_state_conversion.py 
# -i data/maniskill/StackCube-v0/trajectory.h5 -o data/maniskill/zarr
@click.command()
@click.option('-i', '--input', required=True, help='input dir contains npy files')
@click.option('-o', '--output', required=True, help='output zarr path')
@click.option('--abs_action', is_flag=True, default=False)
def main(input, output, abs_action):
    # Load HDF5 containing trajectories
    traj_path = pathlib.Path(input)
    ori_h5_file = h5py.File(traj_path, "r")
    
    # Load associated json
    json_path = str(traj_path).replace(".h5",".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    n_ep=10
    episodes = json_data["episodes"]

    buffer = ReplayBuffer.create_empty_numpy()
    for ind in range(n_ep):
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"

        ori_actions = ori_h5_file[traj_id]["actions"][:]
        ori_env_states = ori_h5_file[traj_id]["env_states"][1:]

        data = {                              
            'state': ori_env_states,
            'action': ori_actions
        }
        buffer.add_episode(data)

    buffer.save_to_path(zarr_path=output, chunk_length=-1)

if __name__ == '__main__':
    main()