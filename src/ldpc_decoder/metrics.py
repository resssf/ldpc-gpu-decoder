"""Decoding metrics and visualization"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class DecodingMetrics:
    """Metrics for LDPC decoding"""

    @staticmethod
    def bit_error_rate(true_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
        """BER"""
        return np.mean(true_bits != decoded_bits)

    @staticmethod
    def frame_error_rate(true_codewords: np.ndarray, decoded_codewords: np.ndarray) -> float:
        """FER"""
        errors = np.sum(np.any(true_codewords != decoded_codewords, axis=1))
        return errors / len(true_codewords)

    @staticmethod
    def throughput_mbps(total_bits: int, time_ns: float) -> float:
        time_s = max(time_ns / 1e9, 1e-9)  # Defeat from zero value
        return (total_bits / time_s) / 1e6  # Mbps

    @staticmethod
    def latency_ms(time_ms: float, num_frames: int) -> float:
        """Per-frame latency"""
        return time_ms / num_frames

class Visualizer:
    """Plotting utilities"""

    @staticmethod
    def plot_ber_vs_snr(results_df: pd.DataFrame, name: str):
        """BER vs SNR"""
        plt.figure(figsize=(10, 6))

        for method in results_df['method'].unique():
            data = results_df[results_df['method'] == method]
            plt.semilogy(data['snr_db'], data['ber'], marker='o', label=method, linewidth=2)

        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('BER', fontsize=12)
        plt.title(f'BER vs SNR ({name})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_throughput_comparison(results_df: pd.DataFrame, name: str):
        """Throughput comparison"""
        plt.figure(figsize=(10, 6))

        cpu_data = results_df[results_df['method'] == 'cpu']
        gpu_data = results_df[results_df['method'] == 'gpu']

        plt.plot(cpu_data['snr_db'], cpu_data['throughput_mbps'], marker='s', label='CPU', linewidth=2)
        plt.plot(gpu_data['snr_db'], gpu_data['throughput_mbps'], marker='^', label='GPU', linewidth=2)

        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.title(f'Throughput Comparison: CPU vs GPU ({name})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_speedup(results_df: pd.DataFrame, name: str):
        """GPU speedup relative to CPU"""
        plt.figure(figsize=(10, 6))

        cpu_data = results_df[results_df['method'] == 'cpu'].set_index('snr_db')
        gpu_data = results_df[results_df['method'] == 'gpu'].set_index('snr_db')

        speedup = gpu_data['throughput_mbps'] / cpu_data['throughput_mbps']

        plt.plot(speedup.index, speedup.values, marker='D', linewidth=2, markersize=8, color='green')
        plt.axhline(y=1, color='red', linestyle='--', label='No speedup')

        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Speedup (GPU / CPU)', fontsize=12)
        plt.title(f'GPU Speedup over CPU ({name})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
