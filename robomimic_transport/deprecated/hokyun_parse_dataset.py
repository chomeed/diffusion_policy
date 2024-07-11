import h5py
import robomimic
import robomimic.utils.file_utils as FileUtils
# the dataset registry can be found at robomimic/__init__.py
from robomimic import DATASET_REGISTRY
import numpy as np

def parse_data_robomimic(dataset_dir, save_dir):
    dataset_path = dataset_dir
    f = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data"
    demos = list(f["data"].keys()) # total, env_args, demo_0, demo_1, ... (unsorted)
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos]) # 0, 1, 2, ... (sorted)
    demos = [demos[i] for i in inds] # demo_0, demo_1, demo_2, ... (sorted)
 
    print("hdf5 file {} has {} demonstrations".format(dataset_path, num_demos))
    for ep in range(num_demos):
        demo_key = demos[ep] # e.g. demo_0
        demo_grp = f["data/{}".format(demo_key)] # e.g. data/demo_0
        print(demo_grp)
        print(demo_grp["obs"].keys())
        print(demo_grp["obs/robot0_joint_pos"].shape)
        print(demo_grp["actions"])
        for k, v in demo_grp["obs"].items(): 
            print(k, v)
        print(demo_grp["obs/robot0_joint_pos"][0])
        
        break
        qpos = np.concatenate((demo_grp["obs/robot0_joint_pos"], demo_grp["obs/robot1_joint_pos"]), axis=1)
        action = demo_grp["actions"]
        image = demo_grp["obs/agentview_image"]

        max_timesteps = len(image)

        data_dict = {
            '/observations/qpos': qpos,
            '/action': action,
            '/observations/images/camera': image,
        }

        with h5py.File(f'{save_dir}/episode_{ep}.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            image.create_dataset('camera', (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'{save_dir}/episode_{ep}.hdf5 - saved')


if __name__ == "__main__":
    parse_data_robomimic('/Users/meeroro/workspace/bimanual_imitation/diffusion_policy/dataset/transport/ph/image_ds.hdf5', 'parsed_data')