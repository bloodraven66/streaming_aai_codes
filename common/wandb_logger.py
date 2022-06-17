import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class WandbLogger():
    def __init__(self, config):
        self.run = wandb.init(
                            reinit=True,
                            name=config.logging.run_name,
                            project=config.logging.project_name,
                            config=config,
                            notes=config.logging.notes,
                            tags=config.logging.tags)

    def log(self, dct):
        wandb.log(dct)

    def plot_mel(self, samples, gnds, lengths, sampleNo=0):
        sample = samples[sampleNo][:int(lengths[sampleNo])]
        gnd = gnds[sampleNo][:int(lengths[sampleNo])]
        fig, ax = plt.subplots(2, 1)
        sns.set_style("ticks")
        ax[0].imshow(gnd.T)
        ax[1].imshow(sample.T)
        ax[0].set_title('ground truth')
        ax[1].set_title('predicted')
        plt.tight_layout()
        wandb.log({"ema": wandb.Image(plt)})

    def summary(self, dct):
        for key in dct:
            wandb.run.summary[key] = dct[key]

    def log_audio(Self, aud):
        wandb.log({"val": wandb.Audio(aud,  sample_rate=16000)})

    def end_run(self):
        self.run.finish()
