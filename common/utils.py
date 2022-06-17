from attrdict import AttrDict
import yaml
import torch
import numpy as np
from trainer import helperTrain, helperTrain_tts, helperTrain_tts_combine, helperTrain_chunk_enc_dec, helperTrain_chunk_enc, helperTrain_chunk_enc_mem, helperTrain_chunk_enc_mem_future
from models import fastspeech, fastspeech_with_ema, fastspeech_enc_dec_chunk, fastspeech_enc_chunk, fastspeech_enc_chunk_mem, fastspeech_enc_chunk_mem_future
from models import lstm_baseline
from trainer import helperTrain_lstm

def read_yaml(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def t_(dataset):
    return torch.from_numpy(np.array(dataset))

def get_loaders(datasets, batch_size, pin_memory=False, shuffle=True):
    dataloaders = []
    for dataset in datasets:
        dataset = torch.utils.data.TensorDataset(t_(dataset['mfcc']),
                                                t_(dataset['ema']),
                                                t_(dataset['phon']),
                                                t_(dataset['dur']),
                                                t_(dataset['emaLengths']),
                                                t_(dataset['speakerID']),
                                                )
        dataloaders.append(torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=int(batch_size), pin_memory=pin_memory))
    return dataloaders

def get_trainer(config):
    if config.common.model == 'aai_transformer':
            return helperTrain
    elif config.common.model == 'aai_enc_dec_chunk_transformer':
            return helperTrain_chunk_enc_dec
    elif config.common.model == 'aai_enc_chunk_transformer':
            return helperTrain_chunk_enc
    elif config.common.model == 'lstm':
            return helperTrain_lstm
    elif config.common.model == 'aai_enc_chunk_mem_transformer':
            return helperTrain_chunk_enc_mem
    elif config.common.model == 'aai_enc_chunk_mem_transformer_future':
            return helperTrain_chunk_enc_mem_future


def get_model(config, change_mask=None, window_size=None):
    modelChoice = config.common.model
    mode = config.common.expmode
    if config.data.subjects == 'all':
        subs = 38
    else:
        subs = config.data.subjects
    if modelChoice == 'aai_transformer':
        model_config = read_yaml('config/fs.yaml')
        model = fastspeech.FastSpeech(n_mel_channels=12,
                                        n_speakers=subs,
                                        **model_config).to(config.common.device)
    elif modelChoice == 'aai_enc_dec_chunk_transformer':
        model_config = read_yaml('config/fs_enc_dec_chunk.yaml')
        model = fastspeech_enc_dec_chunk.FastSpeech(n_mel_channels=12,
                                        n_speakers=subs,
                                        **model_config).to(config.common.device)
    elif modelChoice == 'aai_enc_chunk_mem_transformer':
        model_config = read_yaml('config/fs_enc_mem_chunk.yaml')
        if change_mask is not None:
            model_config.mask_type = change_mask
        if window_size is not None:
            model_config.local_window_size = window_size
            config["common"]["chunk_size"] = window_size

        model = fastspeech_enc_chunk_mem.FastSpeech(n_mel_channels=12,
                                        n_speakers=subs,
                                        **model_config).to(config.common.device)
    elif modelChoice == 'aai_enc_chunk_mem_transformer_future':
        model_config = read_yaml('config/fs_enc_mem_chunk_future.yaml')
        if change_mask is not None:
            model_config.mask_type = change_mask
        if window_size is not None:
            model_config.local_window_size = window_size
            config["common"]["chunk_size"] = window_size
        model = fastspeech_enc_chunk_mem_future.FastSpeech(n_mel_channels=12,
                                        n_speakers=subs,
                                        **model_config).to(config.common.device)

    elif modelChoice == 'lstm':
        model_config = read_yaml('config/lstm_baseline.yaml')
        # if change_mask is not None:
        #     model_config.mask_type = change_mask
        # if window_size is not None:
        #     model_config.local_window_size = window_size
        #     config["common"]["chunk_size"] = window_size
        model = lstm_baseline.LSTM(n_mel_channels=12,
                                        **model_config).to(config.common.device)

    elif modelChoice == 'aai_enc_chunk_transformer':
        model_config = read_yaml('config/fs_enc_chunk.yaml')
        if change_mask is not None:
            model_config.mask_type = change_mask
        if window_size is not None:
            model_config.local_window_size = window_size
            config["common"]["chunk_size"] = window_size
        model = fastspeech_enc_chunk.FastSpeech(n_mel_channels=12,
                                        n_speakers=subs,
                                        **model_config).to(config.common.device)

    else:
        raise Exception('model Not found')


    return model, AttrDict({**model_config, **config})


def load_pretrained(config, model):
    model.load_state_dict(torch.load(config.common.ema_pretrained))
    return model
