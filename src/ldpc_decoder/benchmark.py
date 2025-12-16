"""Benchmark for LDPC codes"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Optional

class LDPCBenchmark:
    """Main orchestrator for benchmarking"""

    def __init__(self, H: csr_matrix = None, code_name: str = 'small_random', max_iterations: int = 10, num_frames_per_snr: int = 10000, sparsity_levels: list = [0.05, 0.1]):
        self.code_name = code_name
        self.max_iterations = max_iterations
        self.num_frames = num_frames_per_snr
        self.sparsity_levels = sparsity_levels
        if H is None:
            if code_name == 'small_random':
                self.H = LDPCCodeGenerator.generate_random_sparse(n=300, k=150, sparsity=0.05)
            elif code_name == 'ieee_1944':
                self.H = LDPCCodeGenerator.ieee_802_11_1944_rate_half()
            elif code_name == '5g_bg1':
                self.H = LDPCCodeGenerator.generate_5g_nr_bg1()
            else:
                raise ValueError("Unknown code_name")
        else:
            self.H = H

        self.m, self.n = self.H.shape
        self.code_rate = (self.n - self.m) / self.n
        self._init_decoders()

    def _init_decoders(self):
        self.cpu_decoder = MinSumDecoderCPU(self.H, self.max_iterations)
        self.gpu_decoder = MinSumDecoderGPU(self.H, self.max_iterations)

    def run(self, snr_dbs: list = None) -> pd.DataFrame:
        if snr_dbs is None:
            snr_dbs = np.arange(-2.0, 10.5, 0.5)
        results = []
        for sparsity in self.sparsity_levels:
            if 'random' in self.code_name:
                self.H = LDPCCodeGenerator.generate_random_sparse(self.n, self.n - self.m, sparsity=sparsity)
                self._init_decoders()
            for snr_db in tqdm(snr_dbs, desc=f"SNR (sparsity={sparsity})"):
                channel = AWGNChannel(snr_db, self.code_rate)
                true_codewords = np.zeros((self.num_frames, self.n), dtype=int)
                received = np.zeros((self.num_frames, self.n), dtype=float)
                for i in range(self.num_frames):
                    received[i] = channel.transmit(true_codewords[i])
                noise_var = 1 / (2 * channel.snr_linear * self.code_rate)
                llrs = np.clip(2 * received / noise_var, -100, 100)
                for method in ['cpu', 'gpu']:
                    try:
                        start_time = time.perf_counter_ns()
                        decoder = self.cpu_decoder if method == 'cpu' else self.gpu_decoder
                        decoded, iters, success = decoder.decode(llrs)
                        end_time = time.perf_counter_ns()
                        time_ns = end_time - start_time
                        ber = DecodingMetrics.bit_error_rate(true_codewords, decoded)
                        fer = DecodingMetrics.frame_error_rate(true_codewords, decoded)
                        throughput = DecodingMetrics.throughput_mbps(self.num_frames * self.n, time_ns)
                        latency = DecodingMetrics.latency_ms(time_ns, self.num_frames)
                        valid = np.all([Validator.method1_syndrome_check(self.H, decoded[i]) for i in range(self.num_frames) if success[i]])
                    except Exception as e:
                        print(f"Error in {method} at SNR {snr_db}: {e}")
                        ber, fer = 1.0, 1.0
                        throughput, latency = 0.0, float('inf')
                        iters, success = 0, np.zeros(self.num_frames, dtype=bool)
                        valid = False
                    results.append({
                        'snr_db': snr_db,
                        'sparsity': sparsity,
                        'method': method,
                        'ber': ber,
                        'fer': fer,
                        'avg_iterations': np.mean(iters) if isinstance(iters, np.ndarray) else iters,
                        'throughput_mbps': throughput,
                        'latency_ms': latency,
                        'success_rate': np.mean(success),
                        'validation_passed': valid
                    })
        df = pd.DataFrame(results)
        print(f"SUMMARY for {self.code_name}")
        for sp in df['sparsity'].unique():
            print(f"Sparsity {sp}:")
            sp_df = df[df['sparsity'] == sp]
            cpu_data = sp_df[sp_df['method'] == 'cpu']
            gpu_data = sp_df[sp_df['method'] == 'gpu']
            print("  CPU:")
            print(f"    - Avg throughput: {cpu_data['throughput_mbps'].mean():.2f} Mbps")
            print(f"    - Min BER: {cpu_data['ber'].min():.2e}")
            print("  GPU:")
            print(f"    - Avg throughput: {gpu_data['throughput_mbps'].mean():.2f} Mbps")
            print(f"    - Min BER: {gpu_data['ber'].min():.2e}")
            speedup = gpu_data['throughput_mbps'].mean() / cpu_data['throughput_mbps'].mean() if cpu_data['throughput_mbps'].mean() > 0 else 0
            print(f"  GPU Speedup: {speedup:.1f}x")
        return df
