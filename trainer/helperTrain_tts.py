import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from common.wandb_logger import WandbLogger
import seaborn as sns
import scipy.stats
import os
from models.fastspeech import mask_from_lens
import torch.nn.functional as F
import librosa
import numpy as np
MAX_WAV_VALUE = 32768.0

class Operate():
    def __init__(self, params):
        self.patience = int(params.earlystopper.patience)
        self.verbose = params.earlystopper.verbose
        self.checkpoint = params.earlystopper.checkpoint
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valMinLoss = np.Inf
        self.delta = float(params.earlystopper.delta)
        self.minRun = int(params.earlystopper.minRun)
        self.numEpochs = int(params.common.numEpochs)
        self.modelName = params.common.model
        self.eloss = {'train':[], 'val':[]}
        self.mloss = {'train':[], 'val':[]}
        self.plotFolder = params.results.plotFolder
        self.config = params
        if (params.common.expmode == 'both' or params.common.expmode == 'ema'):
            self.predEma = True
        else:
            self.predEma = False
        if (params.common.expmode == 'both' or params.common.expmode == 'mel'):
            self.predMel = True
        else:
            self.predMel = False
        self.dur_loss_fn = F.mse_loss
        self.ema_loss_fn = F.mse_loss
        sns.reset_defaults()
        sns.set()
        self.retrievePaths()
        self.printInitInfo()
        self.logger = WandbLogger(params)

    def retrievePaths(self):
        if self.config.data.subjects == '1': self.unrollLossPlot = True
        else: self.unrollLossPlot = False
        infoset = {'both':'pta_tts',
                    'mel':'tts',
                    'ema':'pta'}
        self.outputFolder = os.path.join(self.config.results.plotFolder,
                                        infoset[self.config.common.expmode])
        self.outputPrefix = '_'.join([self.config.common.expdetail,
                                        self.config.common.model,
                                        self.config.common.datasetName,
                                        self.config.data.subjects])


    def printInitInfo(self):
        infoset = {'both':'PTA+TTS',
                    'mel':'TTS',
                    'ema':'PTA'}
        print('='*100)
        print(f'Starting {infoset[self.config.common.expmode]}')
        print('-'*100)
        print(f'Output: {self.outputFolder} | ',end='')
        print(f'Exp prefix: {self.outputPrefix} | ',)
        print('='*100)

    def update(self, mode, eloss, mloss):
        self.eloss[mode].append(eloss)
        self.mloss[mode].append(mloss)

    def esCheck(self):
        score = -self.trackValLoss
        if self.epoch>self.minRun:
            if self.bestScore is None:
                self.bestScore = score
                self.saveCheckpoint()
            elif score < self.bestScore + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.earlyStop = True
            else:
                self.bestScore = score
                self.saveCheckpoint()
                self.counter = 0

    def get_audio(self, sample, lengths):
        y_gen_tst = sample[:int(lengths[0])].T
        y_gen_tst = np.exp(y_gen_tst)
        S = librosa.feature.inverse.mel_to_stft(
                y_gen_tst,
                power=1,
                sr=22050,
                n_fft=1024,
                fmin=0,
                fmax=8000.0)
        audio = librosa.core.griffinlim(
                S,
                n_iter=32,
                hop_length=256,
                win_length=1024)
        print(audio.shape)
        audio = audio * MAX_WAV_VALUE
        audio = audio.astype('int16')
        return audio

    def saveCheckpoint(self):
        if self.verbose:
            print(f'Validation loss decreased ({self.valMinLoss:.6f} --> {self.trackValLoss:.6f}).  Saving model ...')
        torch.save(self.model.state_dict(), self.checkpoint)
        self.valMinLoss = self.trackValLoss

    def plot_losses(self):
        if self.predEma and self.predMel:
            title = 'PTA TTS'
        elif self.predEma:
            title = 'PTA'
        elif self.predMel:
            title = 'TTS'
        fig, ax = plt.subplots(1, 2)
        plt.suptitle(f'model: {title}')
        for idx, key in enumerate(self.eloss):
            if self.predEma:
                ax[idx].plot(self.eloss[key], label=key+'_ema', color='#003366')
                if self.predMel and self.config.common.overallplot:
                    with open(f'outputs/melplotdump_{key}_large.npy', 'rb') as f:
                        onlytts = np.load(f)
                    ax[idx].plot(onlytts[:len(self.eloss[key])], label=key+'_tts', color='black', linestyle='dashed')

            if self.predMel:
                ax[idx].plot(self.mloss[key], label=key+'_mel', color='#336600')
                if not self.predEma:
                    with open(f'outputs/melplotdump_{key}_large.npy', 'wb') as f:
                        np.save(f, np.array(self.mloss[key]))
            ax[idx].legend()
            ax[idx].set_ylim((0, 5))
        plt.tight_layout()

        plt.savefig(os.path.join(self.outputFolder, self.outputPrefix+'loss.png'))
        plt.clf()

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
        plt.savefig(os.path.join(self.outputFolder, 'mels', self.outputPrefix +'mel.png'))
        plt.clf()

    def trainloop(self, loader, mode, break_run=False):
        if mode == 'train': self.model.train()
        elif mode == 'val': self.model.eval()
        else: raise NotImplementedError
        cc, rmse = [], []
        eloss_, mloss_ = 0, 0
        self.trackValLoss = 0
        skipped = 0
        losses_to_upload = {'mel':[], 'dur':[], 'total':[]}
        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):

                mfcc, ema, phon, dur, mel, emaLengths, melLengths, sp_id = self.set_device(data, ignoreList=[0, 5])
                phon_lens = torch.from_numpy(np.array([len([ph_idx for ph_idx in phon_ if ph_idx != 0]) for phon_ in phon])).to(self.config.common.device)
                if not self.config.common.sub_embed:
                    sp_id = None
                try:
                    assert torch.all(torch.round(melLengths/(dur.sum(dim=1)*0.8634)))
                except:
                    skipped += 1
                    continue

                inputs = (phon.long(), phon_lens, mel, melLengths, sp_id.long(), dur)
                emaOut, dur_pred, log_dur_pred = self.model(inputs)
                assert len(emaOut.shape) == 3
                ema = mel[:, :emaOut.shape[-2]:, ]
                eloss, mloss = 0, 0
                # emaLengths = emaLengths.detach().cpu().numpy()
                # emaLengths = [self.config.data.emaPadMax if i > self.config.data.emaPadMax else i for i in emaLengths]
                minLens = []
                log_dur_tgt = torch.log(dur.float() + 1)
                dur_mask = mask_from_lens(phon_lens, max_len=dur.size(1))
                dur_pred_loss = self.dur_loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
                dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()
                lengths_for_ema = []
                for loss_idx in range(len(ema)):
                    ema_index = int(emaLengths[loss_idx])
                    lengths_for_ema.append(ema_index)
                    if loss_idx == 0:
                        ema_loss = self.ema_loss_fn(emaOut[loss_idx, :ema_index, ], ema[loss_idx, :ema_index, ])
                    else:
                        ema_loss += self.ema_loss_fn(emaOut[loss_idx, :ema_index, ], ema[loss_idx, :ema_index, ])
                ema_loss = ema_loss / len(emaOut)
                loss = dur_pred_loss + ema_loss
                losses_to_upload['dur'].append(dur_pred_loss.item())
                losses_to_upload['mel'].append(ema_loss.item())
                losses_to_upload['total'].append(loss.item())
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    self.trackValLoss += loss.item()

                tepoch.set_postfix(loss=loss.item())
                if break_run:
                    break
        print('Num skipped', skipped)
        for key in losses_to_upload:
            self.logger.log({f'{mode}_{key}_loss':sum(losses_to_upload[key])/len(losses_to_upload[key])})
        if mode == 'val':
            self.trackValLoss /= len(loader)
            self.logger.plot_mel(emaOut.detach().cpu().numpy(), ema.detach().cpu().numpy(), melLengths, sampleNo=0)
            aud = self.get_audio(emaOut.detach().cpu().numpy()[0], melLengths)
            self.logger.log_audio(aud)
        del eloss, mloss, loss, mfcc, ema, phon, dur, mel, emaLengths, sp_id
        self.cc = round(np.mean(np.mean(cc, axis=0)), 4)
        self.rmse = round(np.mean(np.mean(rmse, axis=0)), 4)
        return

    def trainer(self, model, loaders):
        trainLoader, valLoader, _ = loaders
        self.optimizer, self.lossFn, self.scheduler = self.get_trainers(model)
        self.model = model
        if not self.config.common.infer:
            if os.path.exists(self.config.earlystopper.checkpoint):
                if not self.config.earlystopper.reuse:
                    print('checkpoint exists, enable reuse if overwriting')
                    exit()
                else:
                    print('checkpoint overwriting')
            for epoch in range(int(self.config.common.numEpochs)):
                self.epoch = epoch
                self.trainloop(trainLoader, 'train')
                self.trainloop(valLoader, 'val')
                if self.scheduler is not None:
                    self.scheduler.step(self.trackValLoss)

                if self.config.common.verbose:
                    if self.predEma:
                        print(f'[cc: {self.cc}]')

                self.esCheck()
                if self.earlyStop:
                    print("Early stopping at epoch ", epoch)
                    break
                self.plot_losses()
            print('Training completed')

        else:
            print('Starting Inference')
            self.model.load_state_dict(torch.load(self.config.earlystopper.checkpoint))
            self.trainloop(valLoader, 'val', break_run=True)



    def get_cc(self, ema_, pred_, test_lens):
        ema_ = ema_.permute(0, 2, 1).numpy()
        pred_ = pred_.permute(0, 2, 1).numpy()
        m = []
        rMSE = []
        for j in range(len(pred_)):
            c  = []
            rmse = []
            for k in range(12):
                c.append(scipy.stats.pearsonr(ema_[j][k][:test_lens[j]], pred_[j][k][:test_lens[j]])[0])
                rmse.append(np.sqrt(np.mean(np.square(np.asarray(pred_[j][k][:test_lens[j]])-np.asarray(ema_[j][k][:test_lens[j]])))))
            m.append(c)
            rMSE.append(rmse)
        return m, rmse

    def get_trainers(self, model):
        self.lossFnEma  = nn.MSELoss()
        if self.config.optimizer.name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.optimizer.lr), weight_decay=float(self.config.optimizer.weightdecay))
        else:
            raise Exception('Optimizer not found')
        if self.config.common.lossfn == 'mse':
            lossFn = nn.MSELoss()
        elif self.config.common.lossfn == 'l1':
            lossFn = nn.L1Loss()
        else:
            raise Exception('Loss function not found')

        if self.config.common.scheduler == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor=0.6)
        else:
            scheduler = None
        return optimizer, lossFn, scheduler

    def stats(data):
        print(torch.max(data), torch.min(data), torch.mean(data))

    def set_device(self, data, ignoreList):

        if isinstance(data, list):
            return [data[i].to(self.config.common.device).float() if i not in ignoreList else data[i] for i in range(len(data))]
        else:
            raise Exception('set device for input not defined')
