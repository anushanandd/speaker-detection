# -*- coding: utf-8 -*-

"""
Time Difference of Arrival for ReSpeaker 8 Mic Array
Circular array with 8 microphones positioned at 45-degree intervals
"""

import collections

import numpy as np

from .element import Element
from .gcc_phat import gcc_phat

SOUND_SPEED = 340.0

# Microphone distance (assuming similar to 4-mic array)
# For an 8-mic circular array, the diameter would be similar to the 4-mic
# Estimate: 0.081m distance between opposite pairs
MIC_DISTANCE_8 = 0.081
MAX_TDOA_8 = MIC_DISTANCE_8 / float(SOUND_SPEED)


class DOA(Element):
    def __init__(self, rate=16000, chunks=50):
        super(DOA, self).__init__()

        self.queue = collections.deque(maxlen=chunks)
        self.sample_rate = rate

        # Define microphone pairs for 8-mic circular array
        # Pairs are opposite microphones (180 degrees apart)
        # Mic layout (assuming circular, 45 degrees apart):
        # 0: 0°, 1: 45°, 2: 90°, 3: 135°, 4: 180°, 5: 225°, 6: 270°, 7: 315°
        # Opposite pairs: 0-4, 1-5, 2-6, 3-7
        self.pair = [[0, 4], [1, 5], [2, 6], [3, 7]]

    def put(self, data):
        self.queue.append(data)

        super(DOA, self).put(data)

    def get_direction(self):
        tau = [0, 0, 0, 0]
        theta = [0, 0, 0, 0]

        buf = b''.join(self.queue)
        buf = np.fromstring(buf, dtype='int16')

        # Process each microphone pair
        for i, v in enumerate(self.pair):
            tau[i], _ = gcc_phat(
                buf[v[0]::8],  # Extract channel v[0] from 8-channel data
                buf[v[1]::8],  # Extract channel v[1] from 8-channel data
                fs=self.sample_rate,
                max_tau=MAX_TDOA_8,
                interp=1
            )
            theta[i] = np.arcsin(np.clip(tau[i] / MAX_TDOA_8, -1, 1)) * 180 / np.pi

        # Find the pair with minimum absolute tau (most reliable)
        min_index = np.argmin(np.abs(tau))

        # Calculate direction based on the most reliable pair
        # Each pair is 45 degrees apart in the circular array
        if (min_index != 0 and theta[min_index - 1] >= 0) or (min_index == 0 and theta[len(self.pair) - 1] < 0):
            best_guess = (theta[min_index] + 360) % 360
        else:
            best_guess = (180 - theta[min_index])

        # Add offset based on which pair detected the sound
        # Each pair represents 45-degree sectors
        best_guess = (best_guess + min_index * 45) % 360

        return best_guess
