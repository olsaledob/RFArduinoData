import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import toml

module_path = os.path.abspath(os.path.join('..', 'Git/RFAnalysis/sta_analysis/sta_analysis'))
if module_path not in sys.path:
    sys.path.append(module_path)

from receptive_field_analysis import RFAnalysis


class UnitProcessor:
    def __init__(self, data_dict, led_data, rec_id=None, config_file='config.toml', verbose=False):
        # Load config
        with open(config_file, "r") as f:
            config = toml.load(f)

        self.dt = config["unit_processor"]["dt"]
        self.save_plots = config["unit_processor"]["save_plots"]
        self.plot_folder = config["unit_processor"]["plot_folder"]
        self.rec_id = rec_id

        self.data_dict = data_dict
        self.led_timestamps = led_data['timestamps'] / 1e6  # convert to seconds
        self.patterns = led_data['patterns']
        
        # Logging configuration
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Detect multiple stimulation blocks
        self.stim_blocks = self.detect_stim_blocks()
        self.valid_timestamps = self.flatten_stim_timestamps()
        self.bin_edges = np.concatenate([self.valid_timestamps, [self.valid_timestamps[-1] + self.dt]])
        
        self.stimulus = None
        self.process_stimulus()

    def detect_stim_blocks(self):
        """Detect non-contiguous blocks of stimulation."""
        dt = self.dt
        ts = self.led_timestamps
        gap_indices = np.where(np.diff(ts) > dt * 1.5)[0]
        start_indices = np.insert(gap_indices + 1, 0, 0)
        end_indices = np.append(gap_indices, len(ts) - 1)
        blocks = [(ts[start], ts[end] + dt) for start, end in zip(start_indices, end_indices)]
        return blocks

    def flatten_stim_timestamps(self):
        """Merge all timestamps from separate stim blocks into one array."""
        ts = []
        dt = self.dt
        all_ts = self.led_timestamps
        for start, end in self.stim_blocks:
            block_mask = (all_ts >= start) & (all_ts < end)
            ts_block = all_ts[block_mask]
            ts.extend(ts_block)
        return np.array(ts)

    def process_stimulus(self):
        """Convert pattern bytes to stimulus frames only for active periods."""
        # Extract stimulus frames for valid timestamps only
        ts_mask = np.isin(self.led_timestamps, self.valid_timestamps)
        stimulus = self.patterns[ts_mask]

        stimulus = np.unpackbits(stimulus.astype(np.uint8), axis=1)
        stimulus = stimulus.reshape(stimulus.shape[0], 16, 16)
        stimulus = stimulus.transpose((1, 2, 0))  # shape (16, 16, N)

        stimulus_left_half = np.flip(stimulus[:, 0:8, :], axis=1)
        stimulus_right_half = np.flip(stimulus[:, 8:16, :], axis=1)
        self.stimulus = np.hstack((stimulus_left_half, stimulus_right_half))

    def process_all_units(self):
        self.logger.info("Processing all units\n")
        for key in self.data_dict.keys():
            self.process_unit(key)

    def process_unit(self, key):
        self.logger.info(f"Processing Unit {key}\n")
        spike_times = self.data_dict[key]

        # Mask spikes inside stimulation blocks
        stimulus_mask = np.zeros_like(spike_times, dtype=bool)
        for start, end in self.stim_blocks:
            stimulus_mask |= (spike_times >= start) & (spike_times < end)

        spikes_in_stim = spike_times[stimulus_mask]
        n_spikes_in_window = len(spikes_in_stim)
        inside_duration = sum(end - start for start, end in self.stim_blocks)

        spikes_outside_stim = spike_times[~stimulus_mask]
        n_spikes_outside_window = len(spikes_outside_stim)
        outside_duration = (max(spike_times) - min(spike_times)) - inside_duration

        if n_spikes_in_window <= 1:
            self.logger.info(f"Skipping {key}: no spikes in stimulus periods\n")
            return

        firing_rate_inside = n_spikes_in_window / inside_duration if inside_duration > 0 else 0
        firing_rate_outside = n_spikes_outside_window / outside_duration if outside_duration > 0 else 0

        # Filtering criteria
        if (n_spikes_in_window + n_spikes_outside_window) < 500:
            self.logger.info(f"{key} skipped. (barely active)\n")
            return
        if abs(firing_rate_outside - firing_rate_inside) < 0.1:
            self.logger.info(f"{key} skipped. (unchanging)\n")
            return
        if firing_rate_outside - firing_rate_inside > 0:
            self.logger.info(f"{key} skipped. (inhibited)\n")
            return

        self.logger.info(
            f"{key} has {n_spikes_outside_window} outside and {n_spikes_in_window} spikes in stimulus windows\n"
            f"Inside FR: {firing_rate_inside:.3f} Hz, Outside FR: {firing_rate_outside:.3f} Hz\n"
            f"Total inside duration: {inside_duration:.3f} s, outside duration: {outside_duration:.3f} s\n"
        )

        # STA/RTA analysis — ONLY stim periods
        spike_train = np.histogram(spikes_in_stim, bins=self.bin_edges)[0]
        filter_empty = np.zeros((16, 16))
        analysis = RFAnalysis(self.stimulus, spike_train, filter_empty, 'CL')
        analysis.calc_sta(center=True)
        analysis.calc_rta(center=True)
        analysis.plot_sta_lags(f'Rec-ID_{self.rec_id}_{key}', save=self.save_plots)

        # Plotting stim vs spikes
        plt.figure(figsize=(8, 3))
        label_added = False
        for start, end in self.stim_blocks:
            block_ts = self.led_timestamps[(self.led_timestamps >= start) & (self.led_timestamps < end)]
            if not label_added:
                plt.vlines(block_ts, -0.1, 1.1, colors='red', alpha=0.2, label='With Stimulus')
                label_added = True
            else:
                plt.vlines(block_ts, -0.1, 1.1, colors='red', alpha=0.2)
        plt.eventplot(spike_times, lineoffsets=0.5, colors='black', linewidths=0.1)
        plt.xlabel('Time (s)')
        plt.title(f"Stimulus vs spikes: {key}")
        plt.legend()
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_folder, f"Rec-ID_{self.rec_id}_{key}_spiketimes.pdf"), dpi=600)
        plt.show()