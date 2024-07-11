import h5py
import robomimic
# import robomimic.utils.file_utils as FileUtils
# the dataset registry can be found at robomimic/__init__.py
# from robomimic import DATASET_REGISTRY
import numpy as np
import torch 
import os 
from typing import Dict, List, Union, Literal
import pickle 
import numpy as np
from tqdm import tqdm 
from bimanual_imitation.const import *
from bimanual_imitation.rotation_transformer import quat_to_rot6d
from bimanual_imitation.utils import   get_parsed_dataset, \
                    get_dataset_from_tfrecord, \
                    decode_image, \
                    create_sample_indices, \
                    preprocess_img, \
                    get_data_stats, \
                    normalize_data, \
                    sample_sequence

# Modified original for diffusion policy 
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



class BimanualImitationImageDataset(torch.utils.data.Dataset):
    '''
    Dataset for bimanual imitation learning from TFRecord files. 

    Image 
    - shape: (T, C, H, W)


    '''
    def __init__(self, 
                 data_dir_path: Union[str, List[str]]=DATA_DIR_PATH, 
                 pred_horizon: int=PRED_HORIZON,
                 obs_horizon: int=OBS_HORIZON, 
                 action_horizon: int=ACTION_HORIZON,
                 control_type: Literal["end-effector", "joint"]="end-effector"):
        
        assert control_type in ["end-effector", "joint"], "Invalid control type." 

        raw_dataset = get_dataset_from_tfrecord(data_dir_path)
        datasets = get_parsed_dataset(raw_dataset, feature_description=FEATURE_DESCRIPTOR, feature2dim=FEATURE2DIM)
        print(f"{len(datasets)} records loaded.")

        # FIXME: variable episode_ends is not used
        episode_ends = []

        new_datasets = {}

        for record_dict in datasets:
            assert record_dict["steps/observation/image"].shape[0] == record_dict["steps/action/left_local_pose"].shape[0], "timestep of one record does not match."
            sequence_length = record_dict["steps/observation/image"].shape[0]
            episode_ends.append(sequence_length)
            
            for key, value in record_dict.items():
                if key not in new_datasets:
                    new_datasets[key] = value
                else: 
                    new_value = np.concatenate([new_datasets[key], value], axis=0)
                    new_datasets[key] = new_value

        
        episode_ends = np.cumsum(episode_ends)
        print("Episode ends: ", episode_ends)

        datasets = new_datasets
        self.dataset = datasets

        # decode each image and preprocess it 
        images = []
        for image in tqdm(datasets["steps/observation/image"][:-1]): 
            img = decode_image(image)
            img = preprocess_img(img) # outputs torch.Tensor of shape (3, 320, 240)
            images.append(img)
        images = np.array(images) # (T, 3, 320, 240)

        # images_1 = []
        # for image in tqdm(datasets["steps/observation/image_1"][:-1]): 
        #     img = decode_image(image)
        #     img = preprocess_img(img) # outputs torch.Tensor of shape (3, 320, 240)
        #     images_1.append(img)
        # images_1 = np.array(images_1) # (T, 3, 320, 240)

        # images_2 = []
        # for image in tqdm(datasets["steps/observation/image_2"][:-1]): 
        #     img = decode_image(image)
        #     img = preprocess_img(img) # outputs torch.Tensor of shape (3, 320, 240)
        #     images_2.append(img)
        # images_2 = np.array(images_2) # (T, 3, 320, 240)

        # FIXME: save pickle file 
        # with open(os.path.join(DATA_DIR_PATH, "000000-images.pkl"), "wb") as f:
        #     pickle.dump(images, f)

        if control_type == "end-effector":
            ### End-effector position control 
            # quat to 6D - index (3, 6)
            left_ee_pos_quat_gripper = datasets["steps/observation/left_end_effector_pos_quat"]
            right_ee_pos_quat_gripper = datasets["steps/observation/right_end_effector_pos_quat"]
            left_local_pose_quat_gripper = datasets["steps/action/left_local_pose_quat"]
            right_local_pose_quat_gripper = datasets["steps/action/right_local_pose_quat"]

            left_ee_pos_quat_gripper = np.concatenate([left_ee_pos_quat_gripper[:, :3],
                                                    quat_to_rot6d(left_ee_pos_quat_gripper[:, 3:7]).astype(np.float32),
                                                        left_ee_pos_quat_gripper[:, [-1]]], axis=-1) # (T, 10)
            right_ee_pos_quat_gripper = np.concatenate([right_ee_pos_quat_gripper[:, :3],
                                                        quat_to_rot6d(right_ee_pos_quat_gripper[:, 3:7]).astype(np.float32),
                                                        right_ee_pos_quat_gripper[:, [-1]]], axis=-1) # (T, 10)
            left_local_pose_quat_gripper = np.concatenate([left_local_pose_quat_gripper[:, :3],
                                                            quat_to_rot6d(left_local_pose_quat_gripper[:, 3:7]).astype(np.float32),
                                                            left_local_pose_quat_gripper[:, [-1]]], axis=-1) # (T, 10)
            right_local_pose_quat_gripper = np.concatenate([right_local_pose_quat_gripper[:, :3],
                                                            quat_to_rot6d(right_local_pose_quat_gripper[:, 3:7]).astype(np.float32),
                                                            right_local_pose_quat_gripper[:, [-1]]], axis=-1) # (T, 10)

            # normalize pose states
            agent_pos = np.concatenate([left_ee_pos_quat_gripper, # actual
                                        right_ee_pos_quat_gripper, 
                                        left_local_pose_quat_gripper, # desired 
                                        right_local_pose_quat_gripper], axis=-1) # (T, 40)
            
            action = np.concatenate([left_local_pose_quat_gripper, # desired 
                                    right_local_pose_quat_gripper], axis=-1) # (T, 20)

        elif control_type == "joint":
            ## Joint position control
            left_joint_states = datasets["steps/observation/left_joint_states"]
            right_joint_states = datasets["steps/observation/right_joint_states"]
            left_rexel_command = datasets["steps/observation/left_rexel_command"]
            right_rexel_command = datasets["steps/observation/right_rexel_command"]
            
            # normalize pose states
            agent_pos = np.concatenate([left_joint_states, # actual
                                        right_joint_states, 
                                        left_rexel_command, # desired 
                                        right_rexel_command], axis=-1) # (T, 28)
            
            action = np.concatenate([left_rexel_command, # desired 
                                     right_rexel_command], axis=-1) # (T, 14)
        

        # create indices 
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1
        )

        # normalize the data
        stats = dict()
        train_data = {"agent_pos": agent_pos, "action": action, "images": images}
        for key, data in train_data.items():
            if key not in ["images", "images_1", "images_2"]:
                stats[key] = get_data_stats(data)
                train_data[key] = normalize_data(data, stats[key])
            

        # TODO: save stats for normalization 
        with open(os.path.join("data/stats.pkl"), "wb") as f:
            pickle.dump(stats, f)

        # for key in train_data.keys():
        #     train_data[key] = torch.Tensor(train_data[key])
        
        self.train_data = train_data
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon

        # for key, value in self.train_data.items(): 
        #     print(type(value))

        
    def __len__(self) -> int:
        return len(self.indices)
    
    def print_items(self): 
        for key, value in self.dataset.items(): 
            print(key, value.shape)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['images'] = nsample['images'][:self.obs_horizon, :]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
        # nsample['action']
        return nsample

    def normalize_data(self): 
        raise NotImplementedError

    def denormalize_data(self):
        raise NotImplementedError
    
    def save(self): 
        # TODO: save the dataset as a pickle file
        raise NotImplementedError
    
    def quat_to_rot6d(self, quat):
        raise NotImplementedError

    def load(self): 
        raise NotImplementedError


# Unit test 
if __name__ == "__main__": 
    import numpy as np 
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    dataset = BimanualImitationImageDataset()
    # assert len(dataset) == 5453, "Length of dataset is not correct."    
    
    print("All tests passed.")

    

    # print(type(dataset[0]["images"]))
    # print(type(dataset[0]["agent_pos"]))
    # print(type(dataset[0]["action"]))
    
    # print(dataset[0].keys())
    # print(dataset[0]["images"].shape)
    # print(dataset[0]["agent_pos"].shape)
    # print(dataset[0]["action"].shape)

    batch_size = 2
    shuffle = True

    # Create a data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # for i, data in enumerate(dataloader):
    #     fig, axes = plt.subplots(1, 2)
    #     print(data["images"].shape)
    #     print(data["images"][0, 0].shape)
    #     axes[0].imshow(data["images"][0, 0].permute(2, 1, 0)) # permute to (W, H, C))
    #     axes[0].axis('off')
    #     axes[1].imshow(data["images"][0, 1].permute(2, 1, 0))
    #     axes[1].axis('off')
    #     plt.show()

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     ax.scatter(data["action"][0, :, 0], data["action"][0, :, 1], data["action"][0, :, 2], c='r', marker='o')
    #     ax.scatter(data["action"][0, :, 8], data["action"][0, :, 9], data["action"][0, :, 10], c='b', marker='o')

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.show() 




    # dataset.print_items()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for i in range(len(dataset)):
    #     obs, action = dataset[i]
    #     left_pose = obs["left_pose"]
    #     x, y, z = left_pose[:3]  # Extract x, y, z coordinates
    #     ax.scatter(x, y, z, c='r', marker='o')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()

    # for i in range(10): 
    #     obs, action = dataset[i]
    #     print(obs["image"].shape)
    #     print(obs["left_pose"])
    #     print(action["left_pose"])
    #     print("===")

    #     # Visualize the image
    #     plt.imshow(obs["image"].transpose(2, 1, 0))
    #     plt.axis('off')
    #     plt.show()
