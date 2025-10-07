import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import toml

module_path = os.path.abspath(os.path.join('..', 'Git/RFAnalysis/sta_analysis/sta_analysis'))
if module_path not in sys.path:
    sys.path.append(module_path)

from receptive_field_analysis import *

class UnitProcessor:
    def __init__(self, data_dict, led_data, rec_id = None, config_file:str='config.toml',  verbose=False):
        with open(config_file, "r") as f:
            config = toml.load(f)
        
        self.dt = config["unit_processor"]["dt"]
        self.save_plots = config["unit_processor"]["save_plots"]
        self.plot_folder = config["unit_processor"]["plot_folder"]
        self.rec_id = rec_id
        
        self.data_dict = data_dict
        self.led_timestamps = led_data['timestamps'] / 1e6
        self.patterns = led_data['patterns']
        
        self.stim_start = self.led_timestamps[0]
        self.stim_end = self.led_timestamps[-1] + self.dt
        self.bin_edges = np.concatenate([self.led_timestamps, [self.led_timestamps[-1] + self.dt]])

        # Configure logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:  # don't re-add handlers if already added
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.stimulus = None
        self.process_stimulus()

    def process_stimulus(self):
        stimulus = self.patterns
        stimulus = np.unpackbits(stimulus.astype(np.uint8), axis=1)
        stimulus = stimulus.reshape(stimulus.shape[0], 16, 16)
        stimulus = stimulus.transpose((1, 2, 0))  # shape (16, 16, N)
        stimulus_left_half = stimulus[:, 0:8, :]
        stimulus_right_half = stimulus[:, 8:16, :]
        stimulus_left_half = np.flip(stimulus_left_half, axis=1)
        stimulus_right_half = np.flip(stimulus_right_half, axis=1)
        self.stimulus = np.hstack((stimulus_left_half, stimulus_right_half))

    def process_all_units(self):
        self.logger.info("Processing all units\n")
        for key in self.data_dict.keys():
            self.process_unit(key)

    def process_unit(self, key):
        self.logger.info(f"Processing Unit {key}\n")
        spike_times = self.data_dict[key]

        spike_train = np.histogram(spike_times, bins=self.bin_edges)[0]
        stimulus_mask = (spike_times >= self.stim_start) & (spike_times < self.stim_end)
        n_spikes_in_window = stimulus_mask.sum()
        no_stimulus_mask = ~stimulus_mask
        n_spikes_outside_window = no_stimulus_mask.sum()

        if n_spikes_in_window <= 1:
            self.logger.info(f"Skipping {key}: no spikes in stimulus period\n")
            return

        firing_rate_inside = n_spikes_in_window / (self.stim_end - self.stim_start)
        firing_rate_outside = n_spikes_outside_window / (max(spike_times) - (self.stim_end - self.stim_start))

        if (n_spikes_outside_window + n_spikes_in_window) < 500:
            self.logger.info(f"{key} skipped. (barely active)\n")
            return

        if abs(firing_rate_outside - firing_rate_inside) < 0.1:
            self.logger.info(f"{key} skipped. (unchanging)\n")
            return

        if firing_rate_outside - firing_rate_inside > 0:
            self.logger.info(f"{key} skipped. (inhibited)\n")
            return

        self.logger.info(
            f"{key} has {n_spikes_outside_window} outside and {n_spikes_in_window} spikes in stimulus window "
            f"(from {self.stim_start:.2f}s to {self.stim_end:.2f}s)\n"
        )
        self.logger.info(
            f"Firing Rate without Stimulus: \t{round(firing_rate_outside, 3)} Hz\n"
            f"Firing Rate with Stimulus: \t{round(firing_rate_inside, 3)} Hz\n"
            f"Stimulus Duration: \t{round((self.stim_end - self.stim_start), 3)} s, "
            f"No-Stimulus Duration: \t{round((max(spike_times) - (self.stim_end - self.stim_start)), 3)} s\n"
        )

        # RFAnalysis and plotting
        filter_empty = np.zeros((16, 16))
        analysis = RFAnalysis(self.stimulus, spike_train, filter_empty, 'CL')
        analysis.calc_sta(center=True)
        analysis.calc_rta(center=True)
        analysis.plot_sta_lags(f'Rec-ID_{self.rec_id}_{key}', save = self.save_plots)

        plt.figure(figsize=(8, 3))
        plt.vlines(self.led_timestamps, 1, 1.1, colors='red', alpha=0.5, label='With Stimulus')
        plt.eventplot(spike_times, lineoffsets=0.5, colors='black', linewidths=0.1)
        plt.xlabel('Time in s')
        plt.title(f"Stimulus vs spikes: {key}")
        plt.legend()
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_folder, f"Rec-ID_{self.rec_id}_{key}_spiketimes.pdf"), dpi = 600)
        plt.show()

    def printData(self):
        print(self.led_timestamps)
        diffs = np.diff(self.led_timestamps)
        for diff in diffs:
            if diff != 0.0625:
                print(diff)
        print(self.patterns)
        