import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os
from models.fastspeech import mask_from_lens
import torch.nn.functional as F
from common.wandb_logger import WandbLogger

class Operate():
    def __init__(self, params):
        self.patience = int(params.earlystopper.patience)
        self.verbose = params.earlystopper.verbose
        self.checkpoint = params.earlystopper.checkpoint
        self.counter = 0
        self.bestScore = None
        self.dynamic_name = params.earlystopper.dynamic_name
        self.earlyStop = False
        self.valMinLoss = np.Inf
        self.delta = float(params.earlystopper.delta)
        self.minRun = int(params.earlystopper.minRun)
        self.numEpochs = int(params.common.numEpochs)
        self.modelName = params.common.model
        self.chunk_size = params.common.chunk_size
        self.eloss = {'train':[], 'val':[]}
        self.mloss = {'train':[], 'val':[]}
        self.plotFolder = params.results.plotFolder
        self.apply_decode_mask = params.common.decode_mask
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
        print(f'Starting AAI streaming')
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
                self.best_val_cc = self.cc
                self.best_val_ar_cc =  self.ar_cc
                # self.best_val_ar_cc_decodenomask =  self.ar_cc_decodenomask
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
                self.best_val_cc = self.cc
                self.best_val_ar_cc = self.ar_cc
                # self.best_val_ar_cc_decodenomask =  self.ar_cc_decodenomask
                self.counter = 0

    def saveCheckpoint(self):
        if self.verbose:
            print(f'Validation loss decreased ({self.valMinLoss:.6f} --> {self.trackValLoss:.6f}).  Saving model ...')
        if self.dynamic_name:
            self.checkpoint = os.path.join('../savedModels', '_'.join([self.config.common.model,
                                            'subs',
                                            self.config.data.subjects,
                                            'chunksize',
                                            str(self.config.common.chunk_size),
                                            'masktype',
                                            self.config.mask_type,
                                            'default_merged_mask',
                                            '.pth']))
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

    def decode(self, loader):
        self.model.eval()
        cc, rmse = [], []
        self.trackValLoss_ar = 0
        with torch.no_grad():
            with tqdm(loader, unit="batch") as tepoch:
                for counter, data in enumerate(tepoch):

                    mfcc, ema, phon, dur, emaLengths, sp_id = self.set_device(data, ignoreList=[2, 3])
                    mfcc = mfcc[:, :int((torch.ceil(max(emaLengths) / self.chunk_size) * self.chunk_size).item())].cpu()
                    for iter_idx in range(0, mfcc.shape[1]//self.chunk_size):
                        chunk = mfcc[:, (iter_idx)*self.chunk_size :(iter_idx+1)*self.chunk_size]
                        if iter_idx == 0:
                            emaOut, dec_out, future = self.model((chunk.to(emaLengths.device).float(),  emaLengths, sp_id.long()), apply_mask=self.apply_decode_mask, dec_cond=None)
                            emaOut = emaOut.cpu()
                            dec_out = dec_out.cpu()
                        else:
                            emaOut_, dec_out, future = self.model((chunk.to(emaLengths.device).float(),  emaLengths, sp_id.long()), apply_mask=self.apply_decode_mask, dec_cond=dec_out)
                            dec_out = dec_out[:, self.chunk_size:self.chunk_size*2, :]
                            emaOut_ = emaOut_[:, self.chunk_size:self.chunk_size*2, :]
                            emaOut_ = emaOut_.cpu()
                            dec_out = dec_out.cpu()
                            emaOut = torch.cat([emaOut, emaOut_], dim=1)
                    assert len(emaOut.shape) == 3
                    ema = ema[:, :emaOut.shape[-2]:, ]
                    eloss, mloss = 0, 0
                    minLens = []
                    lengths_for_ema = []
                    for loss_idx in range(len(ema)):
                        ema_index = int(emaLengths[loss_idx])
                        lengths_for_ema.append(ema_index)
                        if loss_idx == 0:

                            ema_loss = self.ema_loss_fn(emaOut[loss_idx, :ema_index, ].to(ema.device), ema[loss_idx, :ema_index, ])
                        else:
                            ema_loss += self.ema_loss_fn(emaOut[loss_idx, :ema_index, ].to(ema.device), ema[loss_idx, :ema_index, ])

                    ema_loss = ema_loss / len(emaOut)
                    loss = ema_loss

                    self.trackValLoss_ar += loss.item()
                    metrics = self.get_cc(ema.detach().cpu(), emaOut.detach().cpu(), lengths_for_ema)
                    cc.extend(metrics[0])
                    rmse.extend(metrics[1])

                    tepoch.set_postfix(loss=loss.item())
            self.ar_cc = round(np.mean(np.mean(cc, axis=0)), 4)
            self.rmse = round(np.mean(np.mean(rmse, axis=0)), 4)
            self.trackValLoss_ar /= len(loader)
            self.logger.log({f'AR_cc':self.ar_cc})
            self.logger.log({f'AR_rmse':self.rmse})
            self.logger.log({f'AR_loss':self.trackValLoss_ar})
            print(self.ar_cc)
            del eloss, mloss, loss, mfcc, ema, phon, dur, emaLengths, sp_id

        return

    def decode_nomask(self, loader):
        self.model.eval()
        cc, rmse = [], []
        self.trackValLoss_ar_decodenomask = 0
        with torch.no_grad():
            with tqdm(loader, unit="batch") as tepoch:
                for counter, data in enumerate(tepoch):

                    mfcc, ema, phon, dur, emaLengths, sp_id = self.set_device(data, ignoreList=[2, 3])
                    mfcc = mfcc[:, :int((torch.ceil(max(emaLengths) / self.chunk_size) * self.chunk_size).item())].cpu()
                    for iter_idx in range(0, mfcc.shape[1]//self.chunk_size):
                        chunk = mfcc[:, (iter_idx)*self.chunk_size :(iter_idx+1)*self.chunk_size]
                        if iter_idx == 0:
                            emaOut, dec_out = self.model((chunk.to(emaLengths.device).float(),  emaLengths, sp_id.long()), apply_mask=False).cpu()
                        else:
                            emaOut, dec_out = torch.cat([emaOut, self.model((chunk.to(emaLengths.device).float(),  emaLengths, sp_id.long()), apply_mask=False).cpu()], dim=1)
                    assert len(emaOut.shape) == 3
                    ema = ema[:, :emaOut.shape[-2]:, ]
                    eloss, mloss = 0, 0
                    # emaLengths = emaLengths.detach().cpu().numpy()
                    # emaLengths = [self.config.data.emaPadMax if i > self.config.data.emaPadMax else i for i in emaLengths]
                    minLens = []
                    lengths_for_ema = []
                    for loss_idx in range(len(ema)):
                        ema_index = int(emaLengths[loss_idx])
                        lengths_for_ema.append(ema_index)
                        if loss_idx == 0:

                            ema_loss = self.ema_loss_fn(emaOut[loss_idx, :ema_index, ].to(ema.device), ema[loss_idx, :ema_index, ])
                        else:
                            ema_loss += self.ema_loss_fn(emaOut[loss_idx, :ema_index, ].to(ema.device), ema[loss_idx, :ema_index, ])

                    ema_loss = ema_loss / len(emaOut)
                    loss = ema_loss

                    self.trackValLoss_ar_decodenomask += loss.item()
                    metrics = self.get_cc(ema.detach().cpu(), emaOut.detach().cpu(), lengths_for_ema)
                    cc.extend(metrics[0])
                    rmse.extend(metrics[1])

                    tepoch.set_postfix(loss=loss.item())
            self.ar_cc_decodenomask = round(np.mean(np.mean(cc, axis=0)), 4)
            self.rmse = round(np.mean(np.mean(rmse, axis=0)), 4)
            self.trackValLoss_ar_decodenomask /= len(loader)
            self.logger.log({f'AR_cc_decodenomask':self.ar_cc_decodenomask})
            self.logger.log({f'AR_rmse_decodenomask':self.rmse})
            self.logger.log({f'AR_loss_decodenomask':self.trackValLoss_ar_decodenomask})
            print(self.ar_cc_decodenomask)
            del eloss, mloss, loss, mfcc, ema, phon, dur, emaLengths, sp_id

        return
    def trainloop(self, loader, mode):
        if mode == 'train': self.model.train()
        elif mode == 'val': self.model.eval()
        else: raise NotImplementedError
        cc, rmse = [], []
        eloss_, mloss_ = 0, 0
        self.trackValLoss = 0
        skipped = 0
        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):

                mfcc, ema, phon, dur, emaLengths, sp_id = self.set_device(data, ignoreList=[2, 3])
                ema_shifted = torch.cat([ema[:, self.chunk_size:, :], torch.zeros((ema.shape[0], self.chunk_size, ema.shape[-1])).to(ema.device)], dim=1)
                inputs = (mfcc, emaLengths, sp_id.long())
                emaOut, _, future = self.model(inputs)

                assert len(emaOut.shape) == 3
                ema = ema[:, :emaOut.shape[-2]:, ]
                eloss, mloss = 0, 0
                # emaLengths = emaLengths.detach().cpu().numpy()
                # emaLengths = [self.config.data.emaPadMax if i > self.config.data.emaPadMax else i for i in emaLengths]
                minLens = []
                lengths_for_ema = []
                for loss_idx in range(len(ema)):
                    ema_index = int(emaLengths[loss_idx])
                    lengths_for_ema.append(ema_index)
                    if loss_idx == 0:
                        future_loss = self.ema_loss_fn(future[loss_idx, :ema_index, ], ema_shifted[loss_idx, :ema_index, ])
                        ema_loss = self.ema_loss_fn(emaOut[loss_idx, :ema_index, ], ema[loss_idx, :ema_index, ])
                    else:
                        future_loss = self.ema_loss_fn(future[loss_idx, :ema_index, ], ema_shifted[loss_idx, :ema_index, ])
                        ema_loss += self.ema_loss_fn(emaOut[loss_idx, :ema_index, ], ema[loss_idx, :ema_index, ])
                future_loss = future_loss / len(emaOut)
                ema_loss = ema_loss / len(emaOut)
                loss = ema_loss + future_loss
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    self.trackValLoss += loss.item()
                    metrics = self.get_cc(ema.detach().cpu(), emaOut.detach().cpu(), lengths_for_ema)
                    cc.extend(metrics[0])
                    rmse.extend(metrics[1])

                tepoch.set_postfix(loss=loss.item())
        self.cc = round(np.mean(np.mean(cc, axis=0)), 4)
        self.rmse = round(np.mean(np.mean(rmse, axis=0)), 4)
        self.logger.log({f'{mode}_cc':self.cc})
        self.logger.log({f'{mode}_rmse':self.rmse})
        # print(ema.shape)
        # plt.plot(ema[0].cpu().numpy()[:, 0])
        # plt.plot(emaOut[0].detach().cpu().numpy()[:, 0])
        # plt.savefig('demo.png')
        # exit()

        if mode == 'val':


            self.trackValLoss /= len(loader)
            self.logger.log({f'val_{mode}_loss':self.trackValLoss})
        print(self.cc)
        del eloss, mloss, loss, mfcc, ema, phon, dur, emaLengths, sp_id

        return

    def trainer(self, model, loaders):



        trainLoader, valLoader, _ = loaders
        self.optimizer, self.lossFn, self.scheduler = self.get_trainers(model)
        self.model = model
        for epoch in range(int(self.config.common.numEpochs)):
            self.epoch = epoch


            self.trainloop(trainLoader, 'train')
            self.trainloop(valLoader, 'val')
            self.decode(valLoader)
            # self.decode_nomask(valLoader)
            if self.scheduler is not None:
                self.scheduler.step(self.trackValLoss)

            if self.config.common.verbose:
                if self.predEma:
                    print(f'[cc: {self.cc}]')

            self.esCheck()
            if self.earlyStop:
                print("Early stopping at epoch ", epoch)
                break
            # self.plot_losses()
        self.logger.log({'best_val_cc':self.best_val_cc})
        self.logger.log({'best_val_ar_cc':self.best_val_ar_cc})
        # self.logger.log({'best_val_ar_cc_decodenomask':self.ar_cc_decodenomask})
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
