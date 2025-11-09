import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import os
import sys
import argparse
import math

BIGINT_WORDS = 8

def int_to_bigint_np(val):
    bigint_arr = np.zeros(BIGINT_WORDS, dtype=np.uint32)
    for j in range(BIGINT_WORDS):
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def load_target_h160_bin(filename):
    targets_bin = bytearray()
    with open(filename) as f:
        for line in f:
            h160_hex = line.strip()
            if len(h160_hex) == 40:
                try: targets_bin.extend(bytes.fromhex(h160_hex))
                except ValueError: pass
    return targets_bin

def init_secp256k1_constants(mod):
    p_data=np.array([0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF],dtype=np.uint32);const_p_gpu=mod.get_global("const_p")[0];cuda.memcpy_htod(const_p_gpu,p_data)
    n_data=np.array([0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF],dtype=np.uint32);const_n_gpu=mod.get_global("const_n")[0];cuda.memcpy_htod(const_n_gpu,n_data)
    g_x=np.array([0x16F81798,0x59F2815B,0x2DCE28D9,0x029BFCDB,0xCE870B07,0x55A06295,0xF9DCBBAC,0x79BE667E],dtype=np.uint32);g_y=np.array([0xFB10D4B8,0x9C47D08F,0xA6855419,0xFD17B448,0x0E1108A8,0x5DA4FBFC,0x26A3C465,0x483ADA77],dtype=np.uint32);g_z=np.array([1,0,0,0,0,0,0,0],dtype=np.uint32);g_infinity=np.array([False],dtype=np.bool_)
    ecpoint_jac_dtype=np.dtype([('X',np.uint32,BIGINT_WORDS),('Y',np.uint32,BIGINT_WORDS),('Z',np.uint32,BIGINT_WORDS),('infinity',np.bool_)])
    g_jac=np.zeros(1,dtype=ecpoint_jac_dtype)
    g_jac['X'],g_jac['Y'],g_jac['Z'],g_jac['infinity']=g_x,g_y,g_z,g_infinity
    const_G_gpu=mod.get_global("const_G_jacobian")[0];cuda.memcpy_htod(const_G_gpu, g_jac)

def run_precomputation(mod):
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1,1,1))
    cuda.Context.synchronize()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=lambda x: int(x,0), required=True)
    parser.add_argument('--end', type=lambda x: int(x,0), required=True)
    parser.add_argument('--step', type=lambda x: int(x,0), default=1)
    parser.add_argument('--file', default='target.txt')
    parser.add_argument('--keys-per-launch', type=int, default=1<<22)
    args = parser.parse_args()

    try:
        with open('kernel160.cu','r') as f:
            full_cuda_code = f.read()
    except FileNotFoundError:
        print("kernel160.cu not found"); sys.exit(1)

    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11','-arch=sm_75','-O3'])
    init_secp256k1_constants(mod)
    run_precomputation(mod)

    find_hash_kernel = mod.get_function("find_hash_kernel_optimized")

    target_bin = load_target_h160_bin(args.file)
    num_targets = len(target_bin) // 20
    if num_targets == 0:
        print("no targets"); sys.exit(1)

    pinned_targets = cuda.pagelocked_empty(len(target_bin), dtype=np.uint8)
    pinned_targets[:] = np.frombuffer(target_bin, dtype=np.uint8)
    d_targets = cuda.mem_alloc(pinned_targets.nbytes)
    cuda.memcpy_htod(d_targets, pinned_targets)

    d_result = cuda.mem_alloc(32)
    d_found_flag = cuda.mem_alloc(4)
    cuda.memset_d32(d_result, 0, 8)
    cuda.memset_d32(d_found_flag, 0, 1)

    found_flag_host = cuda.pagelocked_empty(1, dtype=np.int32)
    found_flag_host[0] = 0
    result_host = cuda.pagelocked_empty(8, dtype=np.uint32)

    streams = [cuda.Stream() for _ in range(2)]

    total_keys_checked = 0
    start_time = time.time()
    current_start = args.start
    current_end = args.end
    step_np = int_to_bigint_np(args.step)
    step_buf = cuda.mem_alloc(step_np.nbytes)
    cuda.memcpy_htod(step_buf, step_np)

    block_size = 256

    try:
        while found_flag_host[0] == 0 and current_start <= current_end:
            keys_left = (current_end - current_start) // args.step + 1
            keys_this_launch = min(args.keys_per_launch, keys_left)
            if keys_this_launch <= 0: break
            grid_size = (keys_this_launch + block_size - 1) // block_size
            start_key_np = int_to_bigint_np(current_start)
            d_start = cuda.mem_alloc(start_key_np.nbytes)
            cuda.memcpy_htod(d_start, start_key_np)
            s = streams[total_keys_checked % len(streams)]
            found_flag_host[0] = 0
            cuda.memcpy_htod_async(d_found_flag, found_flag_host, stream=s)
            find_hash_kernel.prepare("PPPIPPI")
            find_hash_kernel.prepared_call((grid_size,1,1),(block_size,1,1), d_start, np.uint64(keys_this_launch), step_buf, d_targets, np.int32(num_targets), d_result, d_found_flag, stream=s)
            cuda.memcpy_dtoh_async(found_flag_host, d_found_flag, stream=s)
            s.synchronize()
            if found_flag_host[0] != 0:
                cuda.memcpy_dtoh(result_host, d_result)
                print("FOUND", result_host)
                break
            total_keys_checked += keys_this_launch
            elapsed = time.time() - start_time
            speed = total_keys_checked / elapsed if elapsed>0 else 0
            sys.stdout.write(f"\rTotal:{total_keys_checked:,} keys | {speed:,.2f} k/s")
            sys.stdout.flush()
            current_start += keys_this_launch * args.step
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
