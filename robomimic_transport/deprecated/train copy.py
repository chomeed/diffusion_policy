import torch
import torch.nn as nn 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm
import numpy as np 
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from bimanual_imitation.models import get_resnet, replace_bn_with_gn, ConditionalUnet1D
from bimanual_imitation.logger import WandBLogger
from bimanual_imitation.const import OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON, ACTION_DIM
from bimanual_imitation.dataset import BimanualImitationImageDataset


vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)

# ResNet18 has output dim of 512
vision_feature_dim = 512
lowdim_obs_dim = 40
# observation feature has 512+32=544 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim

noise_pred_net = ConditionalUnet1D(
    input_dim=ACTION_DIM,
    global_cond_dim=obs_dim*OBS_HORIZON
)

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)


dataset = BimanualImitationImageDataset(
        data_dir_path=['/home/chomeed/workspace/bimanual-imitation/data/v1/0.1.0', '/home/chomeed/workspace/bimanual-imitation/data/v2/0.1.0'],
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        control_type='end-effector'
    )

print(f"Dataset size: {len(dataset)}")
logger = WandBLogger(project_name='bimanual-imitation-diff-policy', epochs=1000, batch_size=64)
config = logger.config

# create dataloader
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
        # batch loop
        for nbatch in dataloader:
            # data normalized in dataset
            # device transfer
            nimage = nbatch['images'][:, :OBS_HORIZON].to(device)
            nagent_pos = nbatch['agent_pos'][:, :OBS_HORIZON].to(device)
            naction = nbatch['action'].to(device)

            B = nagent_pos.shape[0]

            image_features = nets['vision_encoder'](
                nimage.flatten(end_dim=1))
            image_features = image_features.reshape(
                *nimage.shape[:2],-1)
            # (B,OBS_HORIZON,D)

            obs_features = torch.cat([nagent_pos, image_features], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, OBS_HORIZON * obs_dim)

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)
        
            # predict the noise residual
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

            if epoch_idx % 500 == 0:
                # FIXME: use relative path
                torch.save(nets.state_dict(), f'/home/meedeum/workspace/bimanual-imitation/ckpts/{epoch_idx}.pt')
                
            epoch_loss.append(loss_cpu)
            epoch_cos_distance.append(cos_distance)
            step += 1
            break 

        tglobal.set_postfix(loss=np.mean(epoch_loss), cos_distance=np.mean(epoch_cos_distance))

logger.finish()
