import torch
import pickle, os
import numpy as np
from PIL import Image
from bimanual_imitation.utils import get_data_stats, normalize_data
from diffusion_policy.robomimic_transport.utils import parse_data_robomimic


class RobomimicImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 task_data_dir: str,  
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        images, agent_pos, action, episode_ends = parse_data_robomimic(task_data_dir) 
        # images (T, 240, 320, 3)

        # Permute image dimensions from (N, H, W, C) to (N, C, H, W)
        # Resize images to 128x128
        # Normalize images to [0, 1]
        images = np.array([np.array(Image.fromarray(img).resize((128, 128))) for img in images], dtype=np.float32)
        images = images / 255.0
        images = np.transpose(images, (0, 3, 1, 2))

        agent_pos = agent_pos.astype(np.float32)
        action = action.astype(np.float32)

        # NOT APPLICABLE FOR ROBOMIMIC
        # for view, image_list in images.items():
        #     images[view] = np.transpose(image_list, (0, 3, 1, 2))
        #     images[view] = np.array(list(map(lambda x: Image.fromarray(x).resize((128, 128)), images[view])))
        #     images[view] = images[view] / 255.0

        
        # NOT APPLICABLE FOR ROBOMIMIC
        # Normalize position and gripper state
        # 6D Rotation matrix is not normalized
        # train_agent_pos = np.concatenate([normalize_data(agent_pos[:, :3], self.stats), 
        #                                     agent_pos[:, 3:9], 
        #                                     normalize_gripper(agent_pos[:, 9:10])], axis=-1)

        # train_action = np.concatenate([normalize_data(action[:, :3], self.stats),
        #                                     action[:, 3:9],
        #                                     normalize_gripper(action[:, 9:10])], axis=-1)

        # Normalize agent pos and action 

        stats = dict()
        train_data = {"agent_pos": agent_pos, "action": action, "image_front": images}
        for key, data in train_data.items():
            if key != "image_front":
                stats[key] = get_data_stats(data)
                train_data[key] = normalize_data(data, stats[key])
    
        with open(os.path.join("/Users/meeroro/workspace/bimanual_imitation/diffusion_policy/robomimic_transport/data/transport/ph/stats.pkl"), "wb") as f:
            pickle.dump(stats, f)


        self.train_data = train_data

        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)
        
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        print("Dataset created successfully")
        print(f'Number of samples: {len(self)}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        nsample['image_front'] = nsample['image_front'][:self.obs_horizon,:]
        # nsample['image_wrist'] = nsample['image_wrist'][:self.obs_horizon,:]
        # nsample['image_overhead'] = nsample['image_overhead'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        return nsample
    
def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# Unit test 
if __name__ == '__main__':
    task_data_dir = '/Users/meeroro/workspace/bimanual_imitation/diffusion_policy/robomimic_transport/data/transport/ph/image_dataset.hdf5'
    dataset = RobomimicImageDataset(task_data_dir, pred_horizon=10, obs_horizon=5, action_horizon=5)
    print(f'Number of samples: {len(dataset)}')
    print(f'Number of images: {len(dataset.train_data["image_front"])}')
    print(f'Number of agent positions: {len(dataset.train_data["agent_pos"])}')
    print(f'Number of actions: {len(dataset.train_data["action"])}')
    print(f'Indices: {dataset.indices}')

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=True, pin_memory=True, persistent_workers=True)
    print("Dataset loaded successfully")

    # Output
    batch = next(iter(dataloader))
    print(f'Batch size: {len(batch)}')
    print(f'Batch keys: {batch.keys()}')
    print(f'Batch image_front shape: {batch["image_front"].shape}')
    print(f'Batch agent_pos shape: {batch["agent_pos"].shape}')
    print(f'Batch action shape: {batch["action"].shape}')
