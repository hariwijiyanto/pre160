import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import os
import sys
import argparse
import threading
from queue import Queue
import multiprocessing as mp

def int_to_bigint_np(val):
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8): 
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def load_target_h160_bin(filename):
    targets_bin = bytearray()
    with open(filename) as f:
        for line in f:
            h160_hex = line.strip()
            if len(h160_hex) == 40:
                try: 
                    targets_bin.extend(bytes.fromhex(h160_hex))
                except ValueError: 
                    pass
    return targets_bin

def init_secp256k1_constants(mod, device_id):
    # Set device context
    cuda.Context.synchronize()
    
    p_data = np.array([0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 
                      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
    const_p_gpu = mod.get_global("const_p")[0]
    cuda.memcpy_htod(const_p_gpu, p_data)
    
    n_data = np.array([0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
                      0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
    const_n_gpu = mod.get_global("const_n")[0]
    cuda.memcpy_htod(const_n_gpu, n_data)
    
    g_x = np.array([0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                   0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E], dtype=np.uint32)
    g_y = np.array([0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                   0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77], dtype=np.uint32)
    g_z = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    g_infinity = np.array([False], dtype=np.bool_)
    
    ecpoint_jac_dtype = np.dtype([('X', np.uint32, 8), ('Y', np.uint32, 8), 
                                 ('Z', np.uint32, 8), ('infinity', np.bool_)])
    g_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    g_jac['X'], g_jac['Y'], g_jac['Z'], g_jac['infinity'] = g_x, g_y, g_z, g_infinity
    
    const_G_gpu = mod.get_global("const_G_jacobian")[0]
    cuda.memcpy_htod(const_G_gpu, g_jac)

def run_precomputation(mod):
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()

class GPUWorker:
    def __init__(self, gpu_id, kernel_code, target_bin, args):
        self.gpu_id = gpu_id
        self.device = cuda.Device(gpu_id)
        self.context = self.device.make_context()
        self.target_bin = target_bin
        self.args = args
        self.total_keys_checked = 0
        self.start_time = time.time()
        
        # Compile kernel for this GPU
        print(f"[GPU {gpu_id}] Mengkompilasi kernel...")
        try:
            self.mod = SourceModule(kernel_code, no_extern_c=False, 
                                  options=['-std=c++11', f'-arch=sm_{self.device.compute_capability()[0]}{self.device.compute_capability()[1]}'])
        except Exception as e:
            print(f"[GPU {gpu_id}] Error kompilasi: {e}")
            raise
        
        # Initialize constants
        init_secp256k1_constants(self.mod, gpu_id)
        
        # Run precomputation
        run_precomputation(self.mod)
        
        # Get kernel function
        self.find_hash_kernel = self.mod.get_function("find_hash_kernel_multi_gpu")
        
        # Allocate device memory
        self.num_targets = len(target_bin) // 20
        self.d_targets = cuda.mem_alloc(len(target_bin))
        cuda.memcpy_htod(self.d_targets, np.frombuffer(target_bin, dtype=np.uint8))
        
        self.d_result = cuda.mem_alloc(32)
        self.d_found_flag = cuda.mem_alloc(4)
        
        print(f"[GPU {gpu_id}] Inisialisasi selesai")

    def run_search(self, current_start, current_end, step, total_gpus, 
                  shared_found_flag, shared_result, iteration):
        try:
            cuda.memset_d32(self.d_result, 0, 8)
            cuda.memset_d32(self.d_found_flag, 0, 1)
            
            temp_current = current_start
            keys_in_window = (current_end - current_start) // step + 1
            keys_processed_in_window = 0
            
            while temp_current <= current_end and not shared_found_flag.value:
                keys_left = (current_end - temp_current) // step + 1
                keys_this_launch = min(self.args.keys_per_launch, keys_left)
                if keys_this_launch <= 0: 
                    break

                start_key_np = int_to_bigint_np(temp_current)
                step_np = int_to_bigint_np(step)

                block_size = 256
                grid_size = (keys_this_launch + block_size - 1) // block_size

                # Launch multi-GPU kernel
                self.find_hash_kernel(
                    cuda.In(start_key_np), 
                    np.uint64(keys_this_launch), 
                    cuda.In(step_np),
                    self.d_targets, 
                    np.int32(self.num_targets),
                    self.d_result, 
                    self.d_found_flag,
                    np.int32(self.gpu_id),  # GPU ID
                    np.int32(total_gpus),   # Total GPUs
                    block=(block_size, 1, 1), 
                    grid=(grid_size, 1)
                )
                self.context.synchronize()

                self.total_keys_checked += keys_this_launch
                keys_processed_in_window += keys_this_launch

                # Check if found
                found_flag_host = np.zeros(1, dtype=np.int32)
                cuda.memcpy_dtoh(found_flag_host, self.d_found_flag)
                
                if found_flag_host[0] == 1:
                    found_privkey_np = np.zeros(8, dtype=np.uint32)
                    cuda.memcpy_dtoh(found_privkey_np, self.d_result)
                    privkey_int = 0
                    for j in range(8): 
                        privkey_int |= int(found_privkey_np[j]) << (32 * j)
                    
                    with shared_found_flag.get_lock():
                        if not shared_found_flag.value:
                            shared_found_flag.value = 1
                            for j in range(8):
                                shared_result[j] = found_privkey_np[j]
                            print(f"\n[GPU {self.gpu_id}] KUNCI DITEMUKAN: {hex(privkey_int)}")
                    return True

                elapsed = time.time() - self.start_time
                speed = self.total_keys_checked / elapsed if elapsed > 0 else 0
                window_progress = 100 * keys_processed_in_window / keys_in_window
                
                progress_str = (f"[GPU {self.gpu_id}] Total: {self.total_keys_checked:,} | "
                              f"Speed: {speed:,.2f} k/s | Window: {window_progress:.1f}% | "
                              f"Key: {hex(temp_current)}")
                sys.stdout.write('\r' + progress_str.ljust(120))
                sys.stdout.flush()

                temp_current += keys_this_launch * step
                
                # Check shared flag
                if shared_found_flag.value:
                    return False

            return False

        except Exception as e:
            print(f"\n[GPU {self.gpu_id}] Error: {e}")
            return False

    def cleanup(self):
        try:
            self.d_targets.free()
            self.d_result.free()
            self.d_found_flag.free()
            self.context.pop()
        except:
            pass

def gpu_worker_process(gpu_id, kernel_code, target_bin, args, 
                      shared_found_flag, shared_result, result_queue):
    worker = None
    try:
        worker = GPUWorker(gpu_id, kernel_code, target_bin, args)
        
        iteration = 0
        current_start = args.start
        current_end = args.end
        
        while not shared_found_flag.value and current_end >= 0:
            if worker.run_search(current_start, current_end, args.step, 
                               args.num_gpus, shared_found_flag, shared_result, iteration):
                break
                
            # Slide window down
            current_start -= 1
            current_end -= 1
            iteration += 1
            
            if current_end < 0:
                break
                
    except Exception as e:
        print(f"\n[GPU {gpu_id}] Process error: {e}")
    finally:
        if worker:
            worker.cleanup()
        result_queue.put((gpu_id, worker.total_keys_checked if worker else 0))

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU HASH160 Brute Forcer')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Kunci privat awal')
    parser.add_argument('--end', type=lambda x: int(x, 0), required=True, help='Kunci privat akhir')
    parser.add_argument('--step', type=lambda x: int(x, 0), default=1, help='Langkah penambahan')
    parser.add_argument('--file', default='target.txt', help='File target HASH160')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah kunci per batch GPU')
    parser.add_argument('--gpus', type=int, default=-1, help='Jumlah GPU yang digunakan (-1 untuk semua)')
    args = parser.parse_args()

    # Deteksi GPU
    num_gpus = cuda.Device.count()
    if args.gpus > 0:
        args.num_gpus = min(args.gpus, num_gpus)
    else:
        args.num_gpus = num_gpus
    
    print(f"[*] Terdeteksi {num_gpus} GPU, menggunakan {args.num_gpus} GPU")
    
    if args.num_gpus == 0:
        print("[!] Tidak ada GPU yang terdeteksi")
        sys.exit(1)

    # Load kernel code
    print("[*] Memuat kernel CUDA...")
    try:
        with open('kernel160.cu', 'r') as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print("[!] File 'kernel160.cu' tidak ditemukan")
        sys.exit(1)

    # Load targets
    target_bin = load_target_h160_bin(args.file)
    num_targets = len(target_bin) // 20
    if num_targets == 0:
        print(f"[!] Tidak ada HASH160 valid di {args.file}")
        sys.exit(1)
    print(f"[*] Berhasil memuat {num_targets} HASH160 target")

    # Shared memory untuk koordinasi antar proses
    shared_found_flag = mp.Value('i', 0)
    shared_result = mp.Array('I', 8)  # 8 uint32 untuk BigInt
    result_queue = mp.Queue()

    # Jalankan worker processes
    processes = []
    start_time = time.time()
    
    print(f"[*] Memulai {args.num_gpus} worker processes...")
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=gpu_worker_process, 
                      args=(gpu_id, kernel_code, target_bin, args,
                           shared_found_flag, shared_result, result_queue))
        p.start()
        processes.append(p)

    try:
        # Monitor progress
        total_keys_all_gpus = 0
        while any(p.is_alive() for p in processes):
            time.sleep(2)
            
            # Collect results dari queue
            while not result_queue.empty():
                gpu_id, keys_checked = result_queue.get_nowait()
                total_keys_all_gpus += keys_checked
            
            elapsed = time.time() - start_time
            speed = total_keys_all_gpus / elapsed if elapsed > 0 else 0
            
            sys.stdout.write(f'\r[*] Total semua GPU: {total_keys_all_gpus:,} keys | '
                           f'Speed: {speed:,.2f} k/s | Found: {shared_found_flag.value}')
            sys.stdout.flush()
            
            if shared_found_flag.value:
                break

        # Tunggu semua proses selesai
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    except KeyboardInterrupt:
        print("\n[*] Menghentikan semua processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()

    # Tampilkan hasil akhir
    if shared_found_flag.value:
        print("\n\n[+] KUNCI PRIVAT DITEMUKAN!")
        privkey_int = 0
        for j in range(8):
            privkey_int |= shared_result[j] << (32 * j)
        print(f"    Kunci Privat: {hex(privkey_int)}")
        
        with open("found.txt", "a") as f:
            f.write(f"Kunci Privat: {hex(privkey_int)}\n")
            f.write(f"Waktu: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        print(f"\n\n[+] Pencarian selesai. Tidak ditemukan.")
        print(f"    Total kunci dicoba: {total_keys_all_gpus:,}")

if __name__ == '__main__':
    main()
