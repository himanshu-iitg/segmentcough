#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Python wraper to detect cough """

import os
import sys

import librosa

sys.path.append(os.path.abspath('./ops'))
sys.path.append(os.path.abspath('./ops/cough_detection'))
from ops.feature_class import features
from ops.detection import classify_cough
from scipy.io import wavfile
import pickle
import argparse


def main(input_file):
    """
    Detect cough in a given audio file
    Inputs:
        input_file: (str) path to audio file
    Outputs:
        result: (float) probability that a given file is a cough
    """
    # data_folder = './sample_recordings'
    model = pickle.load(open(os.path.join('./models', 'cough_classifier'),
        'rb'))
    scaler = pickle.load(open(os.path.join('./models',
        'cough_classification_scaler'), 'rb'))

    # fs, x = wavfile.read(input_file)
    x, fs = librosa.load(input_file, sr=None)
    x = (x * 32767).astype(int)
    prob = classify_cough(x, fs, model, scaler)
    print(f"{input_file} has probability of cough: {prob}")
    return prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Path to input audio file',
                        required=True)
    args = parser.parse_args()
    main(args.input)

