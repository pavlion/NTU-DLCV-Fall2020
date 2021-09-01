# try:
#     import wandbaa

# except ModuleNotFoundError:    

class wandb_fake:
    def init(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
    
    def Image(self, *args, **kwargs):
        pass


wandb = wandb_fake()