import pickle
import os, sys, time
import numpy as np
import scipy.io
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import librosa
from tqdm import tqdm
from data_prep.stft import STFT
from librosa.filters import mel as librosa_mel_fn
import torch
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, rootPath, emaFolder, wavFolder, mfccFolder, alignFolder, startStopFolder,
                phonFile, subjects, store, load, path, testMode, nMFCC, sampleRate, numTestSubs,
                phonPadMax, emaPadMax, phonPadValue, emaPadValue, normEma, normMFCC, phoneSequence,
                removeSil, tphnReduce, nfft, hopLength, winLength, nMels, fMin, fMax, normalizeMel,
                filterLength, stdFrac):
        self.rootPath = rootPath
        self.emaFolder = emaFolder
        self.wavFolder = wavFolder
        self.mfccFolder = mfccFolder
        self.alignFolder = alignFolder
        self.startStopFolder = startStopFolder
        self.phonFile = phonFile
        self.subjects = subjects
        self.store = store
        self.load = load
        self.path = path
        self.testMode = testMode
        self.nMFCC = int(nMFCC)
        self.sampleRate = int(sampleRate)
        self.numTestSubs = int(numTestSubs)
        self.phonPadMax = int(phonPadMax)
        self.emaPadMax = int(emaPadMax)
        self.phonPadValue = int(phonPadValue)
        self.emaPadValue = int(emaPadValue)
        self.normEma = normEma
        self.normMFCC = normMFCC
        self.phoneSequence = phoneSequence
        self.removeSil = removeSil
        self.tphnReduce = tphnReduce
        self.nfft = nfft
        self.hopLength = hopLength
        self.winLength = winLength
        self.nMels = nMels
        self.fMin = fMin
        self.fMax = fMax
        self.normalizeMel = normalizeMel
        self.filterLength = filterLength
        self.stdFrac = stdFrac

    def PhnTimeExp(self, alifiles, begin, end):
        begin = begin/100
        end = end/100
        train_aliF_actual = pd.read_csv(alifiles, header=None)

        if self.phoneSequence == True:
            train_aliF = list(train_aliF[0])
            dur, phones = [], []
            beginTrack, endTrack = 0, 0
            allow = False
            exit_timing = False
            beginDisable = -1
            for idx, phone in enumerate(train_aliF):
                phone = phone.split(' ')
                start = phone[0]
                end_ = phone[1].split('\t')[0]
                phoneme = phone[1].split('\t')[-1]
                if self.removeSil:
                    if float(end_)-float(start) <  begin and beginDisable < 0:
                        beginTrack += float(end_)-float(start)
                    elif float(end_)-float(start) ==  begin:
                        beginTrack += float(end_)-float(start)
                        beginDisable = 1
                    else:
                        allow = True
                        if beginDisable == 1:
                            timing = round(beginTrack - begin + float(end_)-float(start), 4)
                            beginDisable = 2
                        else:
                            timing = round(float(end_)-float(start), 4)
                        if beginTrack + float(end_)-float(start) >= end:
                            timing = round(end - beginTrack, 4)
                            exit_timing = True
                        beginTrack += float(end_)-float(start)
                    if allow:
                        dur.append(timing)
                        phones.append(self.wordToInt[phoneme])
                    if exit_timing:
                        break
                else:
                    dur.append(float(end_)-float(start))
                    phones.append(self.wordToInt[phoneme])

        if self.tphnReduce == True:

            info = list(train_aliF_actual[0].map(lambda x:x.split()))
            durs = [(int(float(info[i][0])*100), int(float(info[i][1])*100), info[i][-1]) for i in range(len(info))]
            actual_durs = durs[:]
            total_dur = durs[-1][1] - durs[0][0]
            start = durs[0][1]
            stop = durs[-1][0]
            begin = round(float(begin)*100)
            end = round(float(end)*100)
            for ph_loop_idx in range(len(durs)):
                if begin <= durs[ph_loop_idx][1]:
                    break
            durs = durs[ph_loop_idx:]
            if durs[0][1] == begin:
                durs[0] = (0, begin, 'sil')
            else:
                durs[0] = (begin, durs[0][1], durs[0][-1])
                durs = [(0, begin, 'sil')] + durs

            start = durs[0][1]
            stop = durs[-1][0]

            if start != begin or durs[0][-1] != 'sil':
                raise Exception('debug!')

            if stop != end or durs[-1][-1] != 'sil':
                for ph_loop_idx in range(len(durs)):
                    if end <= durs[ph_loop_idx][1]:
                        break
                durs = durs[:ph_loop_idx+1]
                durs[-1] = (durs[-1][0], end, durs[-1][-1])
                durs = durs + [(end, end+10, 'sil')]
                stop = durs[-1][0]

            phonemes, phoneme_ids, tphn, tphn_ids, durations = [], [], [], [], []
            for (ph_start, ph_end, ph) in durs[1:-1]:
                phonemes.append(ph)
                phoneme_ids.append(self.wordToInt[ph])
                durations.append(ph_end-ph_start)
                tphn.extend([ph]*(durations[-1]))
                tphn_ids.extend([self.wordToInt[ph]]*(durations[-1]))

            for current_dur in durations:
                    assert current_dur>=1
        # print(phoneme_ids, durations, begin, end)
        # exit()
        # phonemes, phoneme_ids, tphn, tphn_ids, durations
        return phoneme_ids, durations

    def getWavEMAPerFile(self, emaFile, mfccFile, phAlignFile, F, beginEnd):
        audioName = Path(emaFile).stem
        emaMat = scipy.io.loadmat(emaFile);
        emaTemp = emaMat['EmaData'];
        emaTemp = np.transpose(emaTemp)
        if self.normEma:
            emaTemp2 = np.delete(emaTemp, [4,5,6,7,10,11],1)
            meanOfData = np.mean(emaTemp2,axis=0)
            emaTemp2 -= meanOfData
            C = 0.5*np.sqrt(np.mean(np.square(emaTemp2),axis=0))
            ema = np.divide(emaTemp2,C)
        else:
            ema = emaTemp
        [aE,bE] = ema.shape
        eBegin = np.int(beginEnd[0, F]*100)
        eEnd = np.int(beginEnd[1, F]*100)-5
        mfcc = np.load(mfccFile).T
        if self.normMFCC:
            assert mfcc.shape[1] == 13
            mean_G = np.mean(mfcc, axis=0)
            std_G = np.std(mfcc, axis=0)
            mfcc = self.stdFrac*(mfcc-mean_G)/std_G
        timeStepsTrack = eEnd-eBegin
        oneHot, durations = self.PhnTimeExp(phAlignFile, begin=eBegin, end=eEnd)
        return ema[eBegin:eEnd,:], mfcc[eBegin:eEnd,:self.nMFCC], oneHot, durations, audioName, eBegin, eEnd

    def melSpec(self, audFile, eBegin, eEnd):
        y, sr = librosa.load(audFile, sr=self.sampleRate)

        y = y[int(sr*eBegin/100): int(sr*eEnd/100)]
        dur = len(y)/sr
        y = torch.from_numpy(y).unsqueeze(0)
        try:
            assert(torch.min(y.data) >= -1)
        except:
            y = y/torch.min(y.data)
        try:
            assert(torch.max(y.data) <= 1)
        except:
            y = y/torch.max(y.data)

        magnitudes, phases = self.stft.transform(y)
        magnitudes = magnitudes.data
        mel = torch.matmul(self.mel_basis, magnitudes).squeeze()
        # spec = librosa.stft(y=y, n_fft=self.nfft, hop_length=self.hopLength, win_length=self.winLength)
        # mel = self.stft.mel_spectrogram(audio_norm).numpy()
        # spec = np.abs(spec)
        # mel = librosa.feature.melspectrogram(S=spec, sr=self.sampleRate, n_fft=self.nfft, n_mels=self.nMels, fmin=self.fMin, fmax=self.fMax)
        if self.normalizeMel:
            mel = np.clip(mel, a_min=1.e-5, a_max=None)
            mel = np.log(mel)

        return mel.T

    def pad(self, dct, skipPad=['emaLengths', 'melLengths', 'audioName', 'beginEnd', 'speakerID']):
        if isinstance(dct, dict):
            for key in dct:
                if key == 'phon':
                    pad = self.phonPadValue
                    padLen = self.phonPadMax
                elif key == 'dur':
                    pad = self.emaPadValue
                    padLen = self.phonPadMax
                else:
                    pad = self.emaPadValue
                    padLen = self.emaPadMax
                if key not in skipPad:
                    dct[key] = pad_sequences(dct[key], padding='post',maxlen=padLen, dtype='float', value=pad)

        else:
            raise Exception('Padding not defined for unkown datatype')
        return dct

    def __call__(self):
        if self.load == True:
            joints = self.path.split('/')
            loadingPath = os.path.join('/'.join(joints[:-1]), self.subjects+ joints[-1])
            if os.path.exists(loadingPath):
                print(pickle.load(open(loadingPath, "rb" ) )[2]['audioName'][2])
                return  pickle.load(open(loadingPath, "rb" ) )
            else:
                raise Exception('Precomputed folder not found')
        for folder in [self.rootPath, self.startStopFolder, self.phonFile]:
            if not os.path.exists(folder):
                raise Exception(f'{folder} not found')

        self.phonDict = np.load(self.phonFile, allow_pickle=True)
        self.setPhoneme = self.phonDict['phones'].item()
        self.wordToInt = self.phonDict['wti'].item()
        self.intToWord = self.phonDict['itw'].item()
        self.intToWord = {}

        for key in self.wordToInt:
            self.wordToInt[key] += 1
            self.intToWord[self.wordToInt[key]] = key

        if self.subjects not in ['1', '10', 'all']:
            raise Exception(f'{self.subjects} must be 1, 10 or all')
        if self.subjects == '1':
            self.subs = ['AshwinHebbar']
        elif self.subjects == '10':
            self.subs = ['AshwinHebbar', 'Babitha', 'DivyaGR', 'GokulS', 'Harshini', 'Pavan_P', 'Prasad','SriRamya','Varshini', 'Vignesh']
        elif self.subjects == 'all':
            self.subs = sorted(os.listdir(os.path.join(self.rootPath, 'DataBase')))
        if self.testMode == 'unseen':
            self.trainSubs = self.subs[:len(self.subs)-self.numTestSubs]
            self.testSubs = self.subs[:-self.numTestSubs]
        elif self.testMode == 'seen':
            self.trainSubs = self.subs
            self.testSubs = self.subs
        else:
            raise Exception(f'{self.testMode} test mode not found')
        testData, valData, trainData = {}, {}, {}
        for key in ['ema', 'mfcc', 'phon', 'dur', 'emaLengths', 'audioName', 'beginEnd', 'speakerID']:
            testData[key] = []
            valData[key] = []
            trainData[key] = []

        self.stft = STFT(filter_length=self.filterLength,
                                 hop_length=self.hopLength,
                                 win_length=self.winLength,
                                 )
        mel_basis = librosa_mel_fn(
            self.sampleRate, self.filterLength, self.nMels, self.fMin, self.fMax)
        self.mel_basis = torch.from_numpy(mel_basis).float()

        for subIdx, sub in enumerate(self.subs):
            mfccDir = os.path.join(self.rootPath, self.mfccFolder, sub)
            emaDir = os.path.join(self.rootPath, 'DataBase', sub, self.emaFolder)
            wavDir = os.path.join(self.rootPath, 'DataBase', sub, self.wavFolder)
            beginEndDir = os.path.join(self.startStopFolder, sub)
            alignDir = os.path.join(self.rootPath, 'FA_EN_ALL', sub, self.alignFolder)
            emaFiles = sorted(os.listdir(emaDir))
            wavFiles = sorted(os.listdir(wavDir))
            mfccFiles = sorted(os.listdir(mfccDir))
            phAlignfiles = sorted(os.listdir(alignDir))
            startStopFile = os.listdir(beginEndDir)
            startStopMAt = scipy.io.loadmat(os.path.join(beginEndDir, startStopFile[0]))
            beginEnd = startStopMAt['BGEN']
            F=10
            for idx in tqdm(range(460)):
                E_t, M_t, P_t, dur, audioName, eBegin, eEnd = self.getWavEMAPerFile(os.path.join(emaDir, emaFiles[idx]), os.path.join(mfccDir, mfccFiles[idx]), os.path.join(alignDir, phAlignfiles[idx]), idx, beginEnd)
                # print(E_t.shape, M_t.shape, Mel_t.shape, Mel_t.shape[0]/M_t.shape[0])
                # exit()

                if sum(dur) != len(E_t):
                    print(sum(dur), len(E_t))
                    exit()


                if ((idx + F) % 10)==0:
                    if sub in self.testSubs:
                        testData['audioName'].append(str(audioName))
                        testData['ema'].append(E_t)
                        testData['emaLengths'].append(len(E_t))
                        testData['mfcc'].append(M_t)
                        testData['phon'].append(P_t)
                        testData['dur'].append(dur)
                        testData['beginEnd'].append([eBegin, eEnd])
                        testData['speakerID'].append(subIdx)

                elif (((idx+F-1)%10)==0):
                    if sub in self.trainSubs:
                        valData['audioName'].append(audioName)
                        valData['ema'].append(E_t)
                        valData['emaLengths'].append(len(E_t))
                        valData['mfcc'].append(M_t)
                        valData['phon'].append(P_t)
                        valData['dur'].append(dur)
                        valData['beginEnd'].append([eBegin, eEnd])
                        valData['speakerID'].append(subIdx)
                else:
                    if sub in self.trainSubs:
                        trainData['audioName'].append(audioName)
                        trainData['ema'].append(E_t)
                        trainData['emaLengths'].append(len(E_t))
                        trainData['mfcc'].append(M_t)
                        trainData['phon'].append(P_t)
                        trainData['dur'].append(dur)
                        trainData['beginEnd'].append([eBegin, eEnd])
                        trainData['speakerID'].append(subIdx)

        trainData = self.pad(trainData)
        valData = self.pad(valData)
        testData = self.pad(testData)

        if self.store == True:
            joints = self.path.split('/')
            storingPath = os.path.join('/'.join(joints[:-1]), self.subjects+ joints[-1])
            pickle.dump([trainData, valData, testData], open(storingPath, "wb" ))
        return trainData, valData, testData
