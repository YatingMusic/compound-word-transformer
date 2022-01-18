import datetime
import glob
import json
import math
import os
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path
import subprocess

import cog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from midi2audio import FluidSynth
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

sys.path.append("workspace/uncond/cp-linear")
from main_cp import *


class Predictor(cog.Predictor):
    def setup(self):
        """Load model"""
        path_data_root = "dataset/representations/uncond/cp/ailab17k_from-scratch_cp"
        path_dictionary = os.path.join(path_data_root, "dictionary.pkl")
        path_ckpt = "checkpoints"  # path to ckpt dir
        loss = 25  # loss
        name = "loss_" + str(loss)
        path_saved_ckpt = os.path.join(path_ckpt, name + "_params.pt")

        # load
        dictionary = pickle.load(open(path_dictionary, "rb"))
        event2word, word2event = dictionary

        # config
        n_class = []
        for key in event2word.keys():
            n_class.append(len(dictionary[0][key]))

        # init model
        net = TransformerModel(n_class, is_training=False)
        net.cuda()
        net.eval()

        # load model
        print("[*] load model from:", path_saved_ckpt)
        net.load_state_dict(torch.load(path_saved_ckpt))

        self.net = net
        self.word2event = word2event
        self.event2word = event2word
        self.dictionary = dictionary
        # self.fs = FluidSynth()

    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    @cog.input(
        "output_type",
        type=str,
        default="audio",
        options=["audio", "midi"],
        help="Output file type, can be audio or midi",
    )
    def predict(self, seed, output_type):
        """Compute prediction"""
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)

        output_path_midi = Path(tempfile.mkdtemp()) / "output.mid"
        output_path_wav = Path(tempfile.mkdtemp()) / "output.wav"
        output_path_mp3 = Path(tempfile.mkdtemp()) / "output.mp3"
        res = None
        while res is None:
            # because sometimes happens: ValueError: probabilities contain NaN
            try:
                res = self.net.inference_from_scratch(self.dictionary)
            except:
                print("Generation failed... Re-trying")

        write_midi(res, str(output_path_midi), self.word2event)

        if output_type == "audio":
            command_fs = (
                "fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 "
                + str(output_path_midi)
                + " -F "
                + str(output_path_wav)
                + " -r 44100"
            )
            os.system(command_fs)
            # self.fs.midi_to_audio(str(output_path_midi), str(output_path_wav))
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(output_path_wav),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(output_path_mp3),
                ],
            )

            return output_path_mp3

        elif output_type == "midi":
            return output_path_midi
