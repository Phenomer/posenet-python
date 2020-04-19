#!/usr/bin/python3

import numpy as np
import sounddevice as sd
import udpclient

client = udpclient.UDPClient('127.0.0.1', 3939)

def audio_callback(indata, frames, time, status):
    mouse = np.mean(np.abs(indata))
    client.send({'mouse': mouse})

with sd.InputStream(
    channels=1, blocksize=1024, dtype=np.int16,
    samplerate=22050, callback=audio_callback):
    input()

