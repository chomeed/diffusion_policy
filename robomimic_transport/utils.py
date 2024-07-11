import h5py
import numpy as np
from tqdm import tqdm


def parse_data_robomimic(dataset_dir):
    dataset_path = dataset_dir
    f = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data"
    demos = list(f["data"].keys())                          # total, env_args, demo_0, demo_1, ... (unsorted)
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])    # 0, 1, 2, ... (sorted)
    demos = [demos[i] for i in inds]                        # demo_0, demo_1, demo_2, ... (sorted)
 
    print("hdf5 file {} has {} demonstrations".format(dataset_path, num_demos))
    
    episode_ends = []
    images = [] 
    action = []
    agent_pos = [] 

    for ep in tqdm(range(10)):
    # for ep in tqdm(range(num_demos)):
        demo_key = demos[ep] # e.g. demo_0
        demo_grp = f["data/{}".format(demo_key)] # e.g. data/demo_0

        agent_pos_0 = demo_grp["obs/robot0_joint_pos"] # (T, 7)
        agent_pos_1 = demo_grp["obs/robot1_joint_pos"] # (T, 7)
        agent_gripper_0 = demo_grp["obs/robot0_gripper_qpos"] # (T, 2)
        agent_gripper_1 = demo_grp["obs/robot1_gripper_qpos"] # (T, 2)

        agent_pos_per_episode = np.concatenate((agent_pos_0, agent_gripper_0, agent_pos_1, agent_gripper_1), axis=1) 
        # (T, 18)

        actions_per_episode = demo_grp["actions"][()] # [()] for numpy array
        # (T, 14) - OSC_POSE + GRIPPER_STATE (x2)  

        images_per_episode = demo_grp["obs/agentview_image"][()] # [()] for numpy array
        # (T, 240, 320, 3)

        episode_length = agent_pos_per_episode.shape[0]
        
        # print("agent pos per episode", agent_pos_per_episode.shape)
        # print("actions per episode", actions_per_episode.shape)
        # print("images per episode", images_per_episode.shape)
        # print("episode length", episode_length)

        images.append(images_per_episode)
        action.append(actions_per_episode)
        agent_pos.append(agent_pos_per_episode)
        episode_ends.append(episode_length)
    
    images = np.concatenate(images, axis=0)
    action = np.concatenate(action, axis=0)
    agent_pos = np.concatenate(agent_pos, axis=0)
    episode_ends = np.cumsum(episode_ends)  
    print(len(episode_ends), "episodes successfully loaded.")
    
    return images, agent_pos, action, episode_ends

# if __name__ == "__main__":
#     task_data_dir = '/Users/meeroro/workspace/bimanual_imitation/diffusion_policy/robomimic_transport/data/transport/ph/image_dataset.hdf5'
#     parse_data_robomimic(task_data_dir)