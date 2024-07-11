from models import get_resnet, replace_bn_with_gn, ConditionalUnet1D
import torch
import torch.nn as nn 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from const import *

from tqdm import tqdm
import numpy as np 
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from logger import WandBLogger

from diffusion_policy.robomimic_transport.dataset import RobomimicImageDataset

vision_encoder_front = get_resnet('resnet18')
vision_encoder_front = replace_bn_with_gn(vision_encoder_front)

vision_feature_dim = 512
lowdim_obs_dim = 18
obs_dim = vision_feature_dim + lowdim_obs_dim

noise_pred_net = ConditionalUnet1D(
    input_dim=ACTION_DIM,
    global_cond_dim=obs_dim*OBS_HORIZON
)

nets = nn.ModuleDict({
    'vision_encoder_front': vision_encoder_front,
    'noise_pred_net': noise_pred_net
})

device = torch.device('cuda')
_ = nets.to(device)

# checkpoint = '/Users/meeroro/workspace/bimanual_imitation/diffusion_policy/robomimic_transport/checkpoints/500.pt'
# try: 
#     nets.load_state_dict(torch.load(checkpoint, map_location=device))
#     print("Checkpoint loaded successfully")
# except:
#     print("Checkpoint not loaded.")

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

logger = WandBLogger(project_name='diffusion-policy-with-robomimic', epochs=1000, batch_size=64)
config = logger.config

dataset = RobomimicImageDataset(
        task_data_dir=DATA_DIR,
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON
    )

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config['batch_size'],
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

num_epochs = config['epochs']

ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

step = 0

with tqdm(range(1, num_epochs+1), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        epoch_cos_distance = list() 

        for nbatch in dataloader:
            nimage_wrist = nbatch['image_wrist'][:,:OBS_HORIZON].to(device)
            nimage_front = nbatch['image_front'][:,:OBS_HORIZON].to(device)
            nimage_overhead = nbatch['image_overhead'][:,:OBS_HORIZON].to(device)
            nagent_pos = nbatch['agent_pos'][:,:OBS_HORIZON].to(device)
            naction = nbatch['action'].to(device)

            B = nagent_pos.shape[0]

            image_features_front = nets['vision_encoder_front'](
                nimage_front.flatten(end_dim=1))
            image_features_front = image_features_front.reshape(
                *nimage_front.shape[:2],-1)
            # (B,OBS_HORIZON,D)

            obs_features = torch.cat([nagent_pos, image_features_front], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, OBS_HORIZON * obs_dim)

            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # Cosine Similarity between predicted noise and GT noise 
            cos = nn.CosineSimilarity(dim=2)
            cos_distance = cos(noise_pred, noise)
            cos_distance = torch.mean(cos_distance).item() 

            # Dot product 
            dot_product = torch.dot(noise.flatten(), noise_pred.flatten()) / noise.shape[0] 

            # norms 
            target_norm = torch.linalg.matrix_norm(noise)
            target_norm = torch.mean(target_norm).item()
            pred_norm = torch.linalg.matrix_norm(noise_pred)
            pred_norm = torch.mean(pred_norm).item()

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())

            # logging
            loss_cpu = loss.item()

            metrics = { 
                "train/train_loss": loss_cpu, 
                "train/cos_distance": cos_distance,
                "train/epoch": epoch_idx + 1,
                "train/target_norm": target_norm,
                "train/predicted_norm": pred_norm, 
                "train/dot_product": dot_product,
            }

            if step % 10 == 0:
                logger.log(metrics, step)

            if epoch_idx % 1000 == 0:
                torch.save(nets.state_dict(), f'/Users/meeroro/workspace/bimanual_imitation/diffusion_policy/robomimic_transport/checkpoints/{epoch_idx}.pt')
                
            epoch_loss.append(loss_cpu)
            epoch_cos_distance.append(cos_distance)
            step += 1
            break 

        tglobal.set_postfix(loss=np.mean(epoch_loss), cos_distance=np.mean(epoch_cos_distance))

logger.finish()
