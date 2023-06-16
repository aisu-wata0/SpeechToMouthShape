
from collections import defaultdict
from dataclasses import dataclass, field

import struct
from typing import Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod

import time
import pickle
from pathlib import Path

import wave
import numpy as np
epsilon = np.finfo(float).eps

import matplotlib.pyplot as plt
import python_speech_features as sf

timers = {}

def logTime(name, elapsed):
    if name not in timers:
        timers[name] = []
    timers[name].append(elapsed)


def timerPrint(keys=None):
    if keys is None:
        keys = timers.keys()
    for k in keys:
        print(k)
        print(np.mean(timers[k]))
        print(np.std(timers[k]))

def save_pickle(obj, filepath):
    """Save a object to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """Load a object from file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

sample_length = 256
trained_model_filepath = "DecisionTreeClassifier_best.pkl"
model = None
try:
    if trained_model_filepath:
        model = load_pickle(trained_model_filepath)
except Exception as e:
    print(e)

mouth_to_class = {
    'a': 0 + 1,
    'i': 1 + 1,
    'u': 2 + 1,
    'e': 3 + 1,
    'o': 4 + 1,
}
# with 0 being no mouth shape

params_normalization = {'mean': {'max': 711112.5, 'min': 999.8803905294758}, 'sd': {'max': 8014158.254359923, 'min': 1286.599240389892}, 'median': {'max': 8354.8828125, 'min': 86.1328125}, 'mode': {'max': 11025.0, 'min': 0.0}, 'Q25': {'max': 6201.5625, 'min': 86.1328125}, 'Q75': {'max': 9991.40625, 'min': 86.1328125}, 'IQR': {'max': 8527.1484375, 'min': 0.0}, 'skew': {'max': 11.0249560469972, 'min': -0.7907742995538356}, 'kurt': {'max': 123.62955768287958, 'min': 0.0}, 'volume_max': {'max': 32767.0, 'min': 0.0}, 'volume_mean': {'max': 19220.188, 'min': 0.0}, 'volume_rms': {'max': 21232.412, 'min': 0.0}, 'freq_max': {'max': 10939.53488372093, 'min': -10939.53488372093}, 'centroid': {'max': 5598.526078816624, 'min': -3297.5241893402645}, 'bandwidth': {'max': 128, 'min': -128}, 'flatness': {'max': 2.0000000000000098, 'min': 0.00015299482225978076}, 'mfcc_0': {'max': 24.233880917366466, 'min': -36.04365338911715}, 'mfcc_1': {'max': 12.554454176163494, 'min': -84.02119233057584}, 'mfcc_2': {'max': 43.675537895753074, 'min': -62.62272546468663}, 'mfcc_3': {'max': 31.907646553322977, 'min': -54.778947796150064}, 'mfcc_4': {'max': 27.373680419554596, 'min': -71.75078161538553}, 'mfcc_5': {'max': 47.012629016512435, 'min': -54.46190636182448}, 'mfcc_6': {'max': 40.16331646319932, 'min': -50.76562755300782}, 'mfcc_7': {'max': 39.48296079407826, 'min': -54.526889816674384}, 'mfcc_8': {'max': 48.395534759571994, 'min': -41.02707561753776}, 'mfcc_9': {'max': 45.00196887359644, 'min': -56.00133514021377}, 'mfcc_10': {'max': 36.10272026258943, 'min': -53.36256356875037}, 'mfcc_11': {'max': 42.23755345027889, 'min': -32.75776462684947}, 'mfcc_12': {'max': 33.15121707347, 'min': -39.15355424906901}, 'logfbank_0': {'max': 15.105728522405887, 'min': -36.04365338911715}, 'logfbank_1': {'max': 15.614791371503687, 'min': -36.04365338911715}, 'logfbank_2': {'max': 18.734701529861006, 'min': -36.04365338911715}, 'logfbank_3': {'max': 19.153880165152078, 'min': -36.04365338911715}, 'logfbank_4': {'max': 19.203046091078264, 'min': -36.04365338911715}, 'logfbank_5': {'max': 20.371671572643574, 'min': -36.04365338911715}, 'logfbank_6': {'max': 20.97016034987541, 'min': -36.04365338911715}, 'logfbank_7': {'max': 21.334005809979175, 'min': -36.04365338911715}, 'logfbank_8': {'max': 21.33885317243855, 'min': -36.04365338911715}, 'logfbank_9': {'max': 21.80236851361376, 'min': -36.04365338911715}, 'logfbank_10': {'max': 22.570830616683985, 'min': -36.04365338911715}, 'logfbank_11': {'max': 23.06864867753392, 'min': -36.04365338911715}, 'logfbank_12': {'max': 21.85895772402514, 'min': -36.04365338911715}, 'logfbank_13': {'max': 21.05722179995083, 'min': -36.04365338911715}, 'logfbank_14': {'max': 22.30122165862296, 'min': -36.04365338911715}, 'logfbank_15': {'max': 22.66529663278015, 'min': -36.04365338911715}, 'logfbank_16': {'max': 22.68310306828238, 'min': -36.04365338911715}, 'logfbank_17': {'max': 23.446048577694928, 'min': -36.04365338911715}, 'logfbank_18': {'max': 23.415488327019535, 'min': -36.04365338911715}, 'logfbank_19': {'max': 22.051496058747805, 'min': -36.04365338911715}, 'logfbank_20': {'max': 22.029513841334353, 'min': -36.04365338911715}, 'logfbank_21': {'max': 21.371627499240898, 'min': -36.04365338911715}, 'logfbank_22': {'max': 21.24705676224204, 'min': -36.04365338911715}, 'logfbank_23': {'max': 21.491085171725285, 'min': -36.04365338911715}, 'logfbank_24': {'max': 21.22217939862436, 'min': -36.04365338911715}, 'logfbank_25': {'max': 20.442105584304212, 'min': -36.04365338911715}, 'ssc_0': {'max': 87.1248285412828, 'min': 44.078062517434276}, 'ssc_1': {'max': 172.98198230500205, 'min': 90.95090830721895}, 'ssc_2': {'max': 298.9609973235045, 'min': 187.59295156440774}, 'ssc_3': {'max': 425.0947936885053, 'min': 268.84035247272163}, 'ssc_4': {'max': 552.430267344236, 'min': 400.2107386101908}, 'ssc_5': {'max': 683.6998328519743, 'min': 529.2072544241851}, 'ssc_6': {'max': 845.4410684369715, 'min': 656.6086527412391}, 'ssc_7': {'max': 1021.1707848426114, 'min': 796.389054566228}, 'ssc_8': {'max': 1190.798648902234, 'min': 971.9530814103304}, 'ssc_9': {'max': 1430.5334059646425, 'min': 1144.949410260746}, 'ssc_10': {'max': 1647.3852460608375, 'min': 1323.2191347299172}, 'ssc_11': {'max': 1900.3619640368936, 'min': 1587.7874403852602}, 'ssc_12': {'max': 2190.789735469687, 'min': 1787.7007042495247}, 'ssc_13': {'max': 2529.829945330722, 'min': 2076.203641711571}, 'ssc_14': {'max': 2890.5162162113425, 'min': 2407.0825254871343}, 'ssc_15': {'max': 3291.4620746264363, 'min': 2761.3931255496555}, 'ssc_16': {'max': 3721.772503858728, 'min': 3197.6062753044826}, 'ssc_17': {'max': 4241.159951637019, 'min': 3614.1449756910315}, 'ssc_18': {'max': 4714.419314728603, 'min': 4135.262136494613}, 'ssc_19': {'max': 5306.347423515851, 'min': 4532.7535513454695}, 'ssc_20': {'max': 5874.757870818822, 'min': 5135.324179182116}, 'ssc_21': {'max': 6602.474263520454, 'min': 5713.952092509708}, 'ssc_22': {'max': 7572.335075148525, 'min': 6592.198348579372}, 'ssc_23': {'max': 8476.800570590758, 'min': 7400.135238134993}, 'ssc_24': {'max': 9421.829037585461, 'min': 8297.637610069454}, 'ssc_25': {'max': 10334.957173988427, 'min': 9226.172624280156}}



def getWavSecondsDurationFromSliceLength(wav, length=None):
    if length == None:
        length = len(wav['data'])
    return length / wav['sample_rate']


def classifyWavFile(filename, model=model):
    wav = getWavData(str(filename))
    return getMouthShapeFromWavData(wav)


def getWavData(filename):
    with wave.open(filename, 'rb') as wav:
        # Get info from wav file
        num_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        seconds = num_frames / sample_rate

        # Read entire wav file
        wav_data = np.frombuffer(wav.readframes(num_frames), dtype=np.int16)
        return {
            'data': wav_data,
            'num_channels': num_channels,
            'sample_rate': sample_rate,
            'num_frames': num_frames,
            'seconds': seconds,
        }

def test_sample(
    filename = "speech.wav", model=model
):
    # start = time.perf_counter()
    wav = getWavData(str(filename))
    # end = time.perf_counter()
    # logTime("getWavData", end - start)
    # start = time.perf_counter()
    mouths, parameter_blocks = getMouthShapeFromWavData(wav)
    # end = time.perf_counter()
    # logTime("getMouthShapeFromWavData", end - start)
    volumes = [r['volume_mean'] for r in parameter_blocks]
    seconds_per_frame = getWavSecondsDurationFromSliceLength(wav, model.info['sample_length'])
    seconds = [seconds_per_frame*i for i in range(len(volumes))]

    fig, ax = plt.subplots()

    # Map mouth shapes to colors
    color_map = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'black'}
    colors = [color_map[shape] for shape in mouths]
    # Plot mouth shapes
    ax.scatter(seconds, volumes, c=colors)
    ax.legend(mouth_to_class.keys())

    # Add volume
    ax.plot(seconds, volumes, 'k-')

    ax.legend()
    fig.tight_layout()
    plt.show()


def getMouthShapeFromWavData(wav, model=model):
    # start = time.perf_counter()
    parameter_blocks = getParams(wav, model.info['sample_length'])
    # end = time.perf_counter()
    # logTime("getParams", end - start)

    # start = time.perf_counter()
    X = prepInput(parameter_blocks, model.info['input_name'], model.info['params_normalization'])
    # end = time.perf_counter()
    # logTime("prepInput", end - start)
    # start = time.perf_counter()

    y_pred = model.predict(X)
    # end = time.perf_counter()
    # logTime("model.predict", end - start)

    # timerPrint()
    return y_pred, parameter_blocks




def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = (spec + epsilon) / (spec.sum() + epsilon)
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / (w ** 3 + epsilon)
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / (w ** 4 + epsilon)

    result_d = {
        'mean': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt
    }

    return result_d

def parameters_from_slice(wav_data, sample_rate, winlen):
    # wav_data is a very short sound sample.
    parameters = {}
    wav_data = wav_data.astype(np.float32)

    # start = time.perf_counter()

    # Volume parameters
    volume_max = np.max(np.abs(wav_data))
    volume_mean = np.mean(np.abs(wav_data))
    argmax = np.argmax(wav_data)
    volume_rms = np.sqrt(np.mean(wav_data**2))
    parameters['volume_max'] = volume_max
    parameters['volume_mean'] = volume_mean
    parameters['volume_rms'] = volume_rms

    parameters = {
        **spectral_properties(wav_data, sample_rate),
        **parameters,
    }

    # Frequency parameters
    spec = np.abs(np.fft.rfft(wav_data))
    fft = spec
    freq = np.fft.fftfreq(len(fft), d=1.0/sample_rate)

    # Frequency parameters
    freq_max = freq[np.argmax(np.abs(fft))]
    parameters['freq_max'] = freq_max
    ## Spectral centroid
    centroid = (np.sum(freq * np.abs(fft)) + epsilon) / (np.sum(np.abs(fft)) + epsilon)
    parameters['centroid'] = centroid
    ## Bandwidth
    bandwidth = np.argmax(np.abs(fft)) - np.argmin(np.abs(fft))
    parameters['bandwidth'] = bandwidth
    ## Spectral flatness
    geometric_mean = np.exp(np.mean(np.log(np.abs(fft) + epsilon)))
    arithmetic_mean = np.mean(np.abs(fft))
    flatness = (geometric_mean +  epsilon) / (arithmetic_mean + epsilon)
    parameters['flatness'] = flatness

    # end = time.perf_counter()
    # logTime("parameters_from_slice", end - start)

    # start = time.perf_counter()

    # MFCCs
    numcep = 13
    
    mfcc = sf.mfcc(wav_data, sample_rate, winlen=winlen, winstep=winlen, numcep=numcep)
    # end = time.perf_counter()
    # logTime("parameters_from_slice.mfcc", end - start)

    # start = time.perf_counter()
    logfbank = sf.logfbank(wav_data, sample_rate, winlen=winlen, winstep=winlen,)
    # end = time.perf_counter()
    # logTime("parameters_from_slice.logfbank", end - start)

    # start = time.perf_counter()
    ssc = sf.base.ssc(wav_data, sample_rate, winlen=winlen, winstep=winlen,)
    # end = time.perf_counter()
    # logTime("parameters_from_slice.ssc", end - start)

    for i in range(numcep):
        parameters[f'mfcc_{i}'] = mfcc[0][i]

    for i in range(len(logfbank[0])):
        parameters[f'logfbank_{i}'] = logfbank[0][i]

    for i in range(len(ssc[0])):
        parameters[f'ssc_{i}'] = ssc[0][i]

    return parameters






def getParams(wav, sample_length=sample_length):
    parameter_blocks = []
    # Loop through audio data by slice
    for i in range(0, wav['num_frames'], sample_length):
        # Get slice of wav data
        slice_data = wav['data'][i:i+sample_length]
        sample_seconds = getWavSecondsDurationFromSliceLength(wav, sample_length)
        params = parameters_from_slice(slice_data, wav['sample_rate'], sample_seconds)
        parameter_blocks.append(params)
    return parameter_blocks



def prepInput(parameter_blocks, input_name, params_normalization):
    tags = input_name.split(',')
    variables = {
        tag: {}
        for tag in tags
    }
    for k in parameter_blocks[0].keys():
        if k in ['mouth_shapes'] or k.startswith('target'):
            continue
        vars = np.array([b[k] for b in parameter_blocks], dtype=np.float64)
        v_max = params_normalization[k]['max']
        v_min = params_normalization[k]['min']
        vars = np.clip((vars - v_min) / (v_max - v_min), 0.0, 1.0)

        if 'X' in tags:
            variables['X'][k] = vars
        if 'X^2' in tags:
            variables['X^2'][k] = vars ** 2
        elif 'X^2' in tags:
            variables['X^3'][k] = vars ** 3
        elif 'X^4' in tags:
            variables['X^4'][k] = vars ** 4

    return varTagsToArray(variables, tags)

def varTagsToArray(variables, tags):
    data = []
    for tag in tags:
        data += [variables[tag][k] for k in variables[tag].keys()]

    return np.array(data).T

def getVariables(parameter_blocks):
    variables = {}
    for k in parameter_blocks[0].keys():
        if k in ['mouth_shapes'] or k.startswith('target'):
            continue
        vars = np.array([b[k] for b in parameter_blocks], dtype=np.float64)
        v_max = params_normalization[k]['max']
        v_min = params_normalization[k]['min']
        vars = (vars - v_min) / (v_max - v_min)

        if 'X' not in variables:
            variables['X'] = {}
        variables['X'][k] = vars

        if 'X^2' not in variables:
            variables['X^2'] = {}
        variables['X^2'][k] = vars ** 2

        if 'X^3' not in variables:
            variables['X^3'] = {}
        variables['X^3'][k] = vars ** 3

        if 'X^4' not in variables:
            variables['X^4'] = {}
        variables['X^4'][k] = vars ** 4
    return variables




def train():
    import matplotlib.pyplot as plt
        
    def getXs(variables):
        Xs = {
            input_name: varTagsToArray(variables, input_name.split(','))
            for input_name in ['X,X^2']
        }
        return Xs


    def arrStarAndEnd(arr, threshold):
        over_idxs = np.where(arr >= threshold)[0]
        # Find start index
        start_idx = over_idxs[0]
        # Find end index
        end_idx = over_idxs[-1]
        return start_idx, end_idx
    
    
    def getSimpleSampleParameters(wav, threshold, threshold_param_name='volume_max', sample_length=sample_length):
        parameter_blocks = getParams(wav, sample_length)
        parameter_blocks = parameter_blocks[:-1]

        idx_start, idx_end = arrStarAndEnd(np.array([
            p[threshold_param_name] for p in parameter_blocks
        ]), threshold)

        for idx, p in enumerate(parameter_blocks):
            if idx < idx_start or idx > idx_end:
                p['target'] = 0
            else:
                p['target'] = 1

        return parameter_blocks

    dataset_folder_filepath = "S:/AI/Voice/tts_clipboard_server/soundfiles/dataset"

    vowel_files = {'a': [], 'e': [], 'i': [], 'o': [], 'u': []}
    file_count = 0
    for dir_character in Path(dataset_folder_filepath).iterdir():
        print(dir_character.name)
        if dir_character.is_dir():
            character_files = {}
            for dir_length in dir_character.iterdir():
                length = dir_length.name
                print(length)
                # filenames = [f for f in dir_length.iterdir() if f.suffix == '.wav']
                filenames = dir_length.glob("*.wav")

                vowel_index = 0
                for filename in sorted(filenames):
                    # vowel_files[list('aeiou')[vowel_index]].append(filename)
                    vowel_files[list('aiueo')[vowel_index]].append(filename)
                    vowel_index = (vowel_index + 1) % 5

                    file_count += 1

        #         break
        #     break
        # break

    print(file_count)

    vowel_params_list = {'a': [], 'e': [], 'i': [], 'o': [], 'u': []}

    for vowel, filenames in vowel_files.items():
        for filename in filenames:
            print(filename)
            wav = getWavData(str(filename))
            parameter_blocks = getSimpleSampleParameters(wav, 200)

            for parameter_block in parameter_blocks:
                for param_name, param in parameter_block.items():
                    if param_name in ['mouth_shapes'] or param_name.startswith('target'):
                        continue
                    if param_name not in params_normalization:
                        params_normalization[param_name] = {
                            'max': param,
                            'min': param,
                        }
                    else:
                        params_normalization[param_name]['max'] = max(param, params_normalization[param_name]['max'])
                        params_normalization[param_name]['min'] = min(param, params_normalization[param_name]['min'])

            vowel_params_list[vowel].append(parameter_blocks)

    print("params_normalization", params_normalization)

    mouth_sounds = list(vowel_params_list.keys())
    class_neg = 0
    class_pos = 1


    dataset = []

    for vowel, vowel_params_list in vowel_params_list.items():
        for parameter_blocks in vowel_params_list:
            targets = {
                sound: np.array([
                    class_pos  if b['target'] and sound == vowel else class_neg
                    for b in parameter_blocks
                ])
                for sound in mouth_sounds
            }

            targets_ = {
                f"targets_{metric_name}": {
                    sound: np.array([
                        b[metric_name] / params_normalization[metric_name]['max'] if b['target'] and sound == vowel else 0.0
                        for b in parameter_blocks
                    ])
                    for sound in mouth_sounds
                }
                for metric_name in ['volume_mean']
            }

            targets_class ={
                'all': np.array([
                    mouth_to_class[vowel] if b['target'] else class_neg
                    # np.array([
                    #     class_pos if b['target'] and sound == vowel else class_neg
                    #     for sound in mouth_sounds
                    # ])
                    for b in parameter_blocks
                ])
            }
            dataset.append({
                'parameter_blocks': parameter_blocks,
                'Xs': getXs(getVariables(parameter_blocks)),
                'targets': targets,
                'targets_class': targets_class,
                **targets_,
            })


    target_arrays = {
        target_name: {
            sound: np.concatenate([d[target_name][sound] for d in dataset])
            for sound in mouth_sounds
        }
        for target_name in [t for t in dataset[0].keys() if t.startswith('target') and t != 'targets_class']
    }

    target_name = 'targets_class'
    target_arrays[target_name] = {
        'all': np.concatenate([d[target_name]['all'] for d in dataset])
    }

    print("target_arrays[target_name]['all'].shape")
    print(target_arrays[target_name]['all'].shape)

    Xs = {}
    for x_name in dataset[0]['Xs'].keys():
        ll = []
        for d in dataset:
            ll.append(d['Xs'][x_name])
        Xs[x_name] = np.concatenate(ll)
        print(Xs[x_name].shape)


    from sklearn import linear_model
    from sklearn import tree
    from sklearn import svm
    from sklearn import model_selection
    from sklearn import metrics
    
    models = {
        target_type: {
            sound: {
                input_name: {}
                for input_name in Xs.keys()
            }
            for sound in [*mouth_sounds, 'all']
        }
        for target_type in target_arrays.keys()
    }

    # Fit models
    for target_type in target_arrays.keys():
        for sound, target in target_arrays[target_type].items():
            y = np.array(target_arrays[target_type][sound])
            for input_name, X in Xs.items():
                if target_type != 'targets_class':
                    models[target_type][sound][input_name]["LinearRegression"] = linear_model.LinearRegression().fit(X, y)
                    models[target_type][sound][input_name]["Lasso"] = linear_model.Lasso(alpha=0.1).fit(X, y)
                    if target_type == 'target':
                        models[target_type][sound][input_name]["RidgeClassifier"] = linear_model.RidgeClassifier().fit(X, y)
                if target_type == 'targets_class':
                    print(f"X {X.shape}", flush=True)
                    print(f"y {y.shape}", flush=True)

                    models[target_type][sound][input_name]["DecisionTreeClassifier"] = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.0).fit(X, y)

                    dot_data = tree.export_graphviz(models[target_type][sound][input_name]["DecisionTreeClassifier"], out_file="DecisionTreeClassifier.dot")


                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)

                    clf = tree.DecisionTreeClassifier(random_state=0)
                    print("Total Impurity vs effective alpha for training set", flush=True)
                    time.sleep(0.001)
                    path = clf.cost_complexity_pruning_path(X_train, y_train)
                    ccp_alphas, impurities = path.ccp_alphas, path.impurities
                    fig, ax = plt.subplots()
                    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
                    ax.set_xlabel("effective alpha")
                    ax.set_ylabel("total impurity of leaves")
                    ax.set_title("Total Impurity vs effective alpha for training set")

                    print("Making clfs", flush=True)
                    print(len(ccp_alphas), flush=True)
                    time.sleep(0.001)
                    clfs = []
                    ccp_alphas_s = [0.0]
                    tolerance_percent = 0.02
                    for ccp_alpha in ccp_alphas:
                        if ccp_alpha > ccp_alphas_s[-1] * (1 + tolerance_percent):
                            ccp_alphas_s.append(ccp_alpha)

                    ccp_alphas = np.array(ccp_alphas_s[:-int(len(ccp_alphas_s) * 0.2)])
                    print(len(ccp_alphas), flush=True)
                    print(ccp_alphas, flush=True)

                    for ccp_alpha in ccp_alphas:
                        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                        clf.fit(X_train, y_train)
                        clfs.append(clf)
                    print(
                        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                            clfs[-1].tree_.node_count, ccp_alphas[-1]
                        )
                    )
                    clfs = clfs[:-1]
                    ccp_alphas = ccp_alphas[:-1]

                    print("Number of nodes vs alpha")
                    time.sleep(0.001)
                    node_counts = [clf.tree_.node_count for clf in clfs]
                    depth = [clf.tree_.max_depth for clf in clfs]
                    fig, ax = plt.subplots(2, 1)
                    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
                    ax[0].set_xlabel("alpha")
                    ax[0].set_ylabel("number of nodes")
                    ax[0].set_title("Number of nodes vs alpha")
                    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
                    ax[1].set_xlabel("alpha")
                    ax[1].set_ylabel("depth of tree")
                    ax[1].set_title("Depth vs alpha")
                    fig.tight_layout()
                    print("node_counts", node_counts, flush=True)
                    print("depth", depth, flush=True)

                    print("Metrics vs alpha for training and testing sets")
                    time.sleep(0.001)
                    fig, ax = plt.subplots()
                    ax.set_xlabel("alpha")
                    ax.set_ylabel("metrics")
                    ax.set_title("Metrics vs alpha for training and testing sets")

                    # ax.plot(ccp_alphas, [clf.score(X_train, y_train) for clf in clfs], marker="o", label="score train", drawstyle="steps-post")
                    # ax.plot(ccp_alphas, [clf.score(X_test, y_test) for clf in clfs], marker="o", label="score test", drawstyle="steps-post")
                    print("[clf.score(X_train, y_train) for clf in clfs]", [clf.score(X_train, y_train) for clf in clfs])
                    print("[clf.score(X_test, y_test) for clf in clfs]",  [clf.score(X_test, y_test) for clf in clfs])
                    y_pred_trains = [clf.predict(X_train) for clf in clfs]
                    y_pred_tests = [clf.predict(X_test) for clf in clfs]
                    m = {}
                    argsorts = {}
                    ax.plot(ccp_alphas, [metrics.precision_score(y_train, y_pred_train, average='macro') for y_pred_train in y_pred_trains], marker="o", label="precision train macro", drawstyle="steps-post")
                    m['precision test macro'] = [metrics.precision_score(y_test, y_pred_test, average='macro') for y_pred_test in y_pred_tests]
                    argsorts['precision test macro'] = np.argsort(m['precision test macro'])[::-1]
                    print(m['precision test macro'])
                    print("argsorts['precision test macro']", argsorts['precision test macro'])
                    print("Best", np.argmax(m['precision test macro']), ccp_alphas[np.argmax(m['precision test macro'])], "precision test macro")
                    ax.plot(ccp_alphas, m['precision test macro'], marker="x", label="precision test macro", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.precision_score(y_train, y_pred_train, average='micro') for y_pred_train in y_pred_trains], marker="o", label="precision train micro", drawstyle="steps-post")
                    m['precision test micro'] = [metrics.precision_score(y_test, y_pred_test, average='micro') for y_pred_test in y_pred_tests]
                    argsorts['precision test micro'] = np.argsort(m['precision test micro'])[::-1]
                    print(m['precision test micro'])
                    print("argsorts['precision test micro']", argsorts['precision test micro'])
                    print("Best", np.argmax(m['precision test micro']), ccp_alphas[np.argmax(m['precision test micro'])], "precision test micro")
                    ax.plot(ccp_alphas, m['precision test micro'], marker="x", label="precision test micro", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.precision_score(y_train, y_pred_train, average='weighted') for y_pred_train in y_pred_trains], marker="o", label="precision train weighted", drawstyle="steps-post")
                    m['precision test weighted'] = [metrics.precision_score(y_test, y_pred_test, average='weighted') for y_pred_test in y_pred_tests]
                    argsorts['precision test weighted'] = np.argsort(m['precision test weighted'])[::-1]
                    print(m['precision test weighted'])
                    print("argsorts['precision test weighted']", argsorts['precision test weighted'])
                    print("Best", np.argmax(m['precision test weighted']), ccp_alphas[np.argmax(m['precision test weighted'])], "precision test weighted")
                    ax.plot(ccp_alphas, m['precision test weighted'], marker="x", label="precision test weighted", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.recall_score(y_train, y_pred_train, average='macro') for y_pred_train in y_pred_trains], marker="o", label="recall train macro", drawstyle="steps-post")
                    m['recall test macro'] = [metrics.recall_score(y_test, y_pred_test, average='macro') for y_pred_test in y_pred_tests]
                    argsorts['recall test macro'] = np.argsort(m['recall test macro'])[::-1]
                    print(m['recall test macro'])
                    print("argsorts['recall test macro']", argsorts['recall test macro'])
                    print("Best", np.argmax(m['recall test macro']), ccp_alphas[np.argmax(m['recall test macro'])], "recall test macro")
                    ax.plot(ccp_alphas, m['recall test macro'], marker="x", label="recall test macro", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.recall_score(y_train, y_pred_train, average='micro') for y_pred_train in y_pred_trains], marker="o", label="recall train micro", drawstyle="steps-post")
                    m['recall test micro'] = [metrics.recall_score(y_test, y_pred_test, average='micro') for y_pred_test in y_pred_tests]
                    argsorts['recall test micro'] = np.argsort(m['recall test micro'])[::-1]
                    print(m['recall test micro'])
                    print("argsorts['recall test micro']", argsorts['recall test micro'])
                    print("Best", np.argmax(m['recall test micro']), ccp_alphas[np.argmax(m['recall test micro'])], "recall test micro")
                    ax.plot(ccp_alphas, m['recall test micro'], marker="x", label="recall test micro", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.recall_score(y_train, y_pred_train, average='weighted') for y_pred_train in y_pred_trains], marker="o", label="recall train weighted", drawstyle="steps-post")
                    m['recall test weighted'] = [metrics.recall_score(y_test, y_pred_test, average='weighted') for y_pred_test in y_pred_tests]
                    argsorts['recall test weighted'] = np.argsort(m['recall test weighted'])[::-1]
                    print(m['recall test weighted'])
                    print("argsorts['recall test weighted']", argsorts['recall test weighted'])
                    print("Best", np.argmax(m['recall test weighted']), ccp_alphas[np.argmax(m['recall test weighted'])], "recall test weighted")
                    ax.plot(ccp_alphas, m['recall test weighted'], marker="x", label="recall test weighted", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.f1_score(y_train, y_pred_train, average='macro') for y_pred_train in y_pred_trains], marker="o", label="f1 train macro", drawstyle="steps-post")
                    m['f1 test macro'] = [metrics.f1_score(y_test, y_pred_test, average='macro') for y_pred_test in y_pred_tests]
                    argsorts['f1 test macro'] = np.argsort(m['f1 test macro'])[::-1]
                    print(m['f1 test macro'])
                    print("argsorts['f1 test macro']", argsorts['f1 test macro'])
                    print("Best", np.argmax(m['f1 test macro']), ccp_alphas[np.argmax(m['f1 test macro'])], "f1 test macro")
                    ax.plot(ccp_alphas, m['f1 test macro'], marker="x", label="f1 test macro", drawstyle="steps-post")

                    ax.plot(ccp_alphas, [metrics.f1_score(y_train, y_pred_train, average='micro') for y_pred_train in y_pred_trains], marker="o", label="f1 train micro", drawstyle="steps-post")
                    m['f1 test micro'] = [metrics.f1_score(y_test, y_pred_test, average='micro') for y_pred_test in y_pred_tests]
                    argsorts['f1 test micro'] = np.argsort(m['f1 test micro'])[::-1]
                    print(m['f1 test micro'])
                    print("argsorts['f1 test micro']", argsorts['f1 test micro'])
                    print("Best", np.argmax(m['f1 test micro']), ccp_alphas[np.argmax(m['f1 test micro'])], "f1 test micro")
                    ax.plot(ccp_alphas, m['f1 test micro'], marker="x", label="f1 test micro", drawstyle="steps-post")

                    ax.legend()
                    fig.tight_layout()

                    # Cross-validate best ones
                    metric_names = [
                        'accuracy',
                        'precision_micro',
                        'precision_macro',
                        'recall_micro',
                        'recall_macro',
                        'f1_micro',
                        'f1_macro',
                    ]
                    metrics_cross = {
                        # metric_name: []
                        # for metric_name in metric_names
                    }
                    idx_cross = argsorts['f1 test macro']
                    ccp_alphas_cross = [ccp_alphas[idx] for idx in idx_cross]
                    # Sort ccp_alphas_cross and f1_scores arrays based on ccp_alphas_cross
                    sort_idx = np.argsort(ccp_alphas_cross)

                    for idx in idx_cross:
                        ccp_alpha = ccp_alphas[idx]
                        print("ccp_alpha", ccp_alpha)
                        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

                        for metric_name in metric_names:
                            scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring=metric_name)
                            print(f"%0.8f {metric_name} with a standard deviation of %0.8f" % (scores.mean(), scores.std()))
                            if  metric_name not in metrics_cross:
                                metrics_cross[metric_name] = []
                            metrics_cross[metric_name].append(scores.mean())
                            if f'{metric_name}_std' not in metrics_cross:
                                metrics_cross[f'{metric_name}_std'] = []
                            metrics_cross[f'{metric_name}_std'].append(scores.std())

                    for metric_name, values in metrics_cross.items():
                        print(metric_name)
                        print(values)


                    metric_get_best = 'f1_macro'
                    best_idx = np.argmax(metrics_cross[metric_get_best])
                    best_ccp_alpha = ccp_alphas_cross[best_idx]
                    # 0.00044728313727107486
                    
                    print(f'best {metric_get_best} {metrics_cross[metric_get_best][best_idx]}')
                    print(f'best_ccp_alpha {best_ccp_alpha}')
                    models[target_type][sound][input_name]["DecisionTreeClassifier_best"] = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=best_ccp_alpha).fit(X, y)

                    models[target_type][sound][input_name]["DecisionTreeClassifier_best"].info = {
                        'sample_length': sample_length,
                        'input_name': input_name,
                        'target_type': target_type,
                        'params_normalization': params_normalization,
                    }
                    save_pickle(models[target_type][sound][input_name]["DecisionTreeClassifier_best"], trained_model_filepath)
                    dot_data = tree.export_graphviz(models[target_type][sound][input_name]["DecisionTreeClassifier_best"], out_file="DecisionTreeClassifier_best.dot")

                    fig, ax = plt.subplots()
                    ax.set_xlabel("alpha")
                    ax.set_ylabel("metrics")
                    ax.set_title("Metrics vs alpha (Cross-validation mean)")
                    for metric_name, values in metrics_cross.items():
                        if metric_name.endswith("_std"):
                            values_t = metrics_cross[metric_name[:-len("_std")]]
                            values_t_sorted = [values_t[idx] for idx in sort_idx]
                            ax.fill_between(
                                [ccp_alphas_cross[idx] for idx in sort_idx],
                                [values_t_sorted[idx] - values[idx] for idx in sort_idx],
                                [values_t_sorted[idx] + values[idx] for idx in sort_idx],
                                alpha=0.2,
                            )
                            continue
                        ax.plot([ccp_alphas_cross[idx] for idx in sort_idx], [values[idx] for idx in sort_idx], marker="x", label=metric_name, drawstyle="steps-post")

                    ax.legend()
                    fig.tight_layout()

                    # accuracy = metrics.accuracy_score(y, y_pred)
                    # cm = metrics.confusion_matrix(y, y_pred)
                    # precision = metrics.precision_score(y, y_pred, average=None)
                    # recall = metrics.recall_score(y, y_pred, average=None)
                    # f1 = metrics.f1_score(y, y_pred, average=None)

                    print(m)
                    print(flush=True)
                    plt.show()


    # evaluations = {
    #     target_type: {
    #         sound: {
    #             'r2_scores': {},
    #             'mae': {},
    #             'mse': {},
    #         } for sound in [*mouth_sounds, 'all']
    #     }
    #     for target_type in target_arrays.keys()
    # }

    # doPlot = True

    # for target_type in target_arrays.keys():
    #     for sound, sound_models in models[target_type].items():
    #         plotted_y = False
    #         if doPlot:
    #             if (target_type != 'targets_class' and sound != 'all'):
    #                 plt.figure()
    #                 plt.title(sound + " " + target_type)
    #                 plt.tight_layout()
    #         for input_name in Xs.keys():
    #             for model_name, model in sound_models[input_name].items():
    #                 label = model_name + input_name
    #                 X = Xs[input_name]
    #                 target = target_arrays[target_type][sound]

    #                 y_pred = model.predict(X)
    #                 y = target

    #                 if (target_type != 'targets_class' and sound != 'all'):
    #                     evaluations[target_type][sound]['r2_scores'][label] = model.score(X, target)
    #                     evaluations[target_type][sound]['mae'][label] = np.mean(np.abs(y_pred - target))
    #                     evaluations[target_type][sound]['mse'][label] = np.mean((y_pred - target)**2)
    #                     print("model_name", label)
    #                     print(f"{sound} R2: {evaluations[target_type][sound]['r2_scores'][label]}")
    #                     print(f"{sound} MAE: {evaluations[target_type][sound]['mae'][label] }")
    #                     print(f"{sound} MSE: {evaluations[target_type][sound]['mse'][label] }")
    #                 else:
    #                     accuracy = metrics.accuracy_score(y, y_pred)
    #                     cm = metrics.confusion_matrix(y, y_pred)
    #                     precision = metrics.precision_score(y, y_pred, average=None)
    #                     recall = metrics.recall_score(y, y_pred, average=None)
    #                     f1 = metrics.f1_score(y, y_pred, average=None)

    #                     print("model_name", label)
    #                     print(f"{sound} accuracy: {accuracy}")
    #                     print(f"{sound} cm: {cm}")
    #                     print(f"{sound} precision: {precision}")
    #                     print(f"{sound} recall: {recall}")
    #                     print(f"{sound} f1: {f1}")

    #                 if doPlot:
    #                     if (target_type != 'targets_class' and sound != 'all'):
    #                         if not plotted_y:
    #                             plt.plot(target, color='b', label='target')
    #                             plotted_y = True
    #                         plt.plot(y_pred, label=label)
    #         if doPlot:
    #             if (target_type != 'targets_class' and sound != 'all'):
    #                 plt.legend()

    # for target_type in target_arrays.keys():
    #     for sound in mouth_sounds:
    #         print(f"Evaluation for {sound} {target_type}:")

    #         r2_scores = evaluations[target_type][sound]['r2_scores']
    #         r2_ranked = sorted(r2_scores, key=r2_scores.get, reverse=True)
    #         print("R2 ranking:", r2_ranked, [r2_scores[x] for x in r2_ranked])

    #         mae = evaluations[target_type][sound]['mae']
    #         mae_ranked = sorted(mae, key=mae.get)
    #         print("MAE ranking:", mae_ranked, [mae[x] for x in mae_ranked])

    #         mse = evaluations[target_type][sound]['mse']
    #         mse_ranked = sorted(mse, key=mse.get)
    #         print("mse ranking:", mse_ranked, [mse[x] for x in mse_ranked])



    # if doPlot:
    #     plt.show()





def speech_test():
    filename = 'speech.wav'
    wav = getWavData(str(filename))
    parameter_blocks = getParams(wav)
    parameter_blocks = parameter_blocks[:-1]


    # # Prepare targets

    target_keyframes = {
        0: None,
        2: "S",
        9: "O",
        13: "N",
        15: "O",
        19: "S",
        24: "S,U",
        28: "B,A",
        32: "I",
        33: "A",
        40: "I",
        42: None,
        43: "E,D",
        44: "E",
        50: "J,U",
        53: "U",
        61: "F",
        62: "F,O",
        64: None,
        71: "K,U",
        75: "S,U",
        78: "W,A",
        88: None,
    }
    max_keyframe = 119

    mouth_sounds = ['S', 'O', 'N', 'A', 'I', 'E', 'F', 'W', 'U', 'D', 'J', 'K', 'B']

    class_neg = 0
    class_pos = 1
    target_arrays = {
        k: [class_neg] * max_keyframe for k in mouth_sounds
    }

    sounds_active = {
        k: class_neg for k in mouth_sounds
    }

    current_shape = None
    for i in range(max_keyframe):
        if i in target_keyframes:
            current_shape = target_keyframes[i]
            sounds_active = {
                k: class_neg for k in mouth_sounds
            }
            if current_shape:
                for s in current_shape.split(","):
                    sounds_active[s] = class_pos
        for s, active in sounds_active.items():
            target_arrays[s][i] = active

    # target_arrays[sound] contains the target array for the equation

    # # Prepare variables


    parameter_blocks_max = {
        k: np.max(np.abs([b[k] for b in parameter_blocks]))
        for k in parameter_blocks[0].keys() if k != 'mouth_shapes'
    }

    print(parameter_blocks_max)


    variables = getVariables(parameter_blocks[:max_keyframe])
    Xs = getXs(variables)

    from sklearn import linear_model
    from sklearn import tree
    from sklearn import svm

    models = {
        sound: {
            input_name: {}
            for input_name in Xs.keys()
            }
        for sound in mouth_sounds
    }
    # Fit linear regression models
    for sound, target in target_arrays.items():
        y = np.array(target_arrays[sound])
        for input_name, X in Xs.items():
            models[sound][input_name]["LinearRegression"] = linear_model.LinearRegression().fit(X, y)
            models[sound][input_name]["Lasso"] = linear_model.Lasso(alpha=0.1).fit(X, y)
            models[sound][input_name]["RidgeClassifier"] = linear_model.RidgeClassifier().fit(X, y)


    evaluations = {
        sound: {
            'r2_scores': {},
            'mae': {},
            'mse': {},
        } for sound in mouth_sounds
    }

    for sound, sound_models in models.items():
        plt.figure()
        plt.title(sound)
        plt.tight_layout()
        plotted_y = False
        for input_name in Xs.keys():
            for model_name, model in sound_models[input_name].items():
                X = Xs[input_name]
                target = target_arrays[sound]
                label = model_name + input_name

                y_pred = model.predict(X)
                evaluations[sound]['r2_scores'][label] = model.score(X, target)

                evaluations[sound]['mae'][label] = np.mean(np.abs(y_pred - target))
                evaluations[sound]['mse'][label] = np.mean((y_pred - target)**2)

                print("model_name", label)
                print(f"{sound} R2: {evaluations[sound]['r2_scores'][label]}")
                print(f"{sound} MAE: {evaluations[sound]['mae'][label] }")
                print(f"{sound} MSE: {evaluations[sound]['mse'][label] }")

                if not plotted_y:
                    plt.plot(target, color='b', label='target')
                    plotted_y = True
                plt.plot(y_pred, label=label)
        plt.legend()

    for sound in mouth_sounds:
        print(f"Evaluation for {sound} sound:")

        r2_scores = evaluations[sound]['r2_scores']
        r2_ranked = sorted(r2_scores, key=r2_scores.get, reverse=True)
        print("R2 ranking:", r2_ranked)

        mae = evaluations[sound]['mae']
        mae_ranked = sorted(mae, key=mae.get)
        print("MAE ranking:", mae_ranked)

        mse = evaluations[sound]['mse']
        mse_ranked = sorted(mse, key=mse.get)
        print("mse ranking:", mse_ranked)


if __name__ == "__main__":
    pass
    # test_sample()
    train()