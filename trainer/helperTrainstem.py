import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os

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
        self.eloss2 = {'train':[], 'val':[]}
        self.mloss2 = {'train':[], 'val':[]}
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
        sns.reset_defaults()
        sns.set()
        self.retrievePaths()
        self.printInitInfo()

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

    def update(self, mode, eloss, mloss, eloss2, mloss2):
        self.eloss[mode].append(eloss)
        self.mloss[mode].append(mloss)
        self.eloss2[mode].append(eloss2)
        self.mloss2[mode].append(mloss2)

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
        self.fig, self.ax = plt.subplots(1, 2)
        plt.suptitle(f'model: {title}')
        for idx, key in enumerate(self.eloss):
            if self.predEma:
                self.ax[idx].plot(self.eloss[key], label=key+'_ema', color='#003366')
                self.ax[idx].plot(self.eloss2[key], label=key+'_emasub', color='yellow')

            if self.predMel:
                self.ax[idx].plot(self.mloss[key], label=key+'_mel', color='#336600')
                self.ax[idx].plot(self.mloss2[key], label=key+'_melsub', color='red')

            self.ax[idx].legend()
            self.ax[idx].set_ylim((0, 5))
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

    def trainloop(self, loader, mode):
        if mode == 'train': self.model.train()
        elif mode == 'val': self.model.eval()
        else: raise NotImplementedError
        cc, rmse = [], []
        eloss_, mloss_ = 0, 0
        self.trackValLoss = 0

        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):
                mfcc, ema, phon, dur, mel, emaLengths = self.set_device(data, ignoreList=[0, 5])
                emaOut, melOut, emaOut2, melOut2 = self.model(phon.float(), durs=dur, lengths=emaLengths)
                eloss, mloss = 0, 0
                eloss2, mloss2 = 0, 0
                emaLengths = emaLengths.detach().cpu().numpy()
                emaLengths = [self.config.data.emaPadMax if i > self.config.data.emaPadMax else i for i in emaLengths]
                minLens = []
                for i in range(len(emaLengths)):
                    if emaOut == None: stopidx = min(int(emaLengths[i]), melOut.shape[1])
                    else: stopidx = min(int(emaLengths[i]), emaOut.shape[1])
                    minLens.append(stopidx)
                    if self.predEma:
                        eloss2 += self.lossFn(emaOut2[i][:stopidx], ema[i][:stopidx])
                        eloss += self.lossFn(emaOut[i][:stopidx], ema[i][:stopidx])
                    if self.predMel:
                        mloss2 += self.lossFn(melOut2[i][:stopidx], mel[i][:stopidx])
                        mloss += self.lossFn(melOut[i][:stopidx], mel[i][:stopidx])
                eloss /= len(emaLengths)
                mloss /= len(emaLengths)
                eloss2 /= len(emaLengths)
                mloss2 /= len(emaLengths)
                if self.predEma and self.predEma:
                    loss = eloss + mloss + eloss2 + mloss2
                elif self.predEma:
                    loss = eloss
                elif self.predMel:
                    loss = mloss
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    self.trackValLoss += loss.item()
                    if self.predEma:
                        metrics = self.get_cc(ema.detach().cpu(), emaOut.detach().cpu(), minLens)
                        cc.extend(metrics[0])
                        rmse.extend(metrics[1])
                if self.unrollLossPlot:
                    if eloss != 0: eloss = eloss.item()
                    if mloss != 0: mloss = mloss.item()
                    if eloss2 != 0: eloss2 = eloss2.item()
                    if mloss2 != 0: mloss2 = mloss2.item()
                    self.update(mode, eloss, mloss, eloss2, mloss2)
                else:
                    if self.predMel:
                        mloss_ += mloss.item()
                    if self.predEma:
                        eloss_ += eloss.item()
                tepoch.set_postfix(loss=loss.item())
        if self.unrollLossPlot == False:
            self.update(mode, eloss_/len(loader), mloss_/len(loader))

        if mode == 'val':
            self.trackValLoss /= len(loader)
            if self.predMel:
                self.plot_mel(melOut.detach().cpu().numpy(), mel.detach().cpu().numpy(), emaLengths)
        del eloss, mloss,eloss2, mloss2, loss, mfcc, ema, phon, dur, mel, emaLengths
        if self.predEma:
            if len(cc)>0:
                self.cc = round(np.mean(np.mean(cc, axis=0)), 4)
                self.rmse = round(np.mean(np.mean(rmse, axis=0)), 4)
        return

    def trainer(self, model, trainLoader, valLoader):
        self.optimizer, self.lossFn, self.scheduler = self.get_trainers(model)
        self.model = model
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
        if self.config.optimizer.name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.optimizer.lr), weight_decay=float(self.config.optimizer.weightdecay))
        else:
            raise Exception('Optimizer not found')
        if self.config.common.lossfn == 'mse':
            lossFn = nn.MSELoss()
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
