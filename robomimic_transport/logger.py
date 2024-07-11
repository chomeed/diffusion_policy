import wandb 


class WandBLogger: 
    def __init__(self, project_name, epochs, batch_size,): 
        wandb.init(
            project=project_name,
            config={"epochs": epochs, "batch_size": batch_size},
            dir="../"
        )
        
        self.config = wandb.config

    def log(self, metrics, step):
        wandb.log(data=metrics, step=step)
     
    def finish(self):
        wandb.finish() 