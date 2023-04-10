import os
import pickle

import librosa
import numpy as np

from ops.detection import classify_cough
from ops.segmentation import segment_cough
from scipy.io import wavfile

BASE_PATH = os.path.abspath(__file__)
COUGH_MODEL = os.path.join(BASE_PATH, '../models/cough_classifier')
COUGH_MODEL_SCALAR = os.path.join(BASE_PATH, '../models/cough_classification_scaler')

with open(COUGH_MODEL, 'rb') as f:
    model = pickle.load(f)

with open(COUGH_MODEL_SCALAR, 'rb') as f:
    scaler = pickle.load(f)


def is_cough_present(fs, x):
    """
    --> detects if cough is present in the voice or not
    :param input_path:
    :return:
    """
    prob = classify_cough(x, fs, model, scaler)
    return prob > 0.8, prob

def segregate_cough(x, fs):
    """
    --> detects if cough is present in the voice or not
    :param input_path:
    :return:
    """
    cough_segments, mask = segment_cough(x, fs)
    return {'segments': [s.tolist() for s in cough_segments], 'mask': mask.tolist()}
    # return {'segments': len(cough_segments)}
    # return str(cough_segments)

def librosa_to_wavfile(s_wave):
    const = 32767
    s_wave = (s_wave * const).astype(int)
    return s_wave

def get_cough(fs: int, x: list):
    """
    --> Divide and segment cough
    """
    x = np.array(x)
    is_cough, prob = is_cough_present(fs, librosa_to_wavfile(x))
    if is_cough:
        data = segregate_cough(x, fs)
        data.update({'cough_prob':prob})
        return
    else:
        return {'segments':[], 'cough_prob': prob}


if __name__ == '__main__':
    path = 'path_to_audio.wav'
    x, fs = librosa.load(path, sr=None)
    print('freq', fs)
    x = x.tolist()
    data = get_cough(fs, x)
    print('len of segments', len(data['segments']))