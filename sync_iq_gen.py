import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import scipy.integrate as trapz

sync_pattern = np.array([
	0x97, 0xfd, 0xd3, 0x7b, 0x0f, 0x1f, 0x6d, 0x08,
	0xf7, 0x83, 0x5d, 0x9e, 0x59, 0x82, 0xc0, 0xfd,
	0x1d, 0xca, 0xad, 0x3b, 0x5b, 0xeb, 0xd4, 0x93,
	0xe1, 0x4a, 0x04, 0xd2, 0x28, 0xdd, 0xf9, 0x01,
	0x53, 0xd2, 0xe6, 0x6c, 0x5b, 0x25, 0x65, 0x31,
	0xc5, 0x7c, 0xe7, 0xf1, 0x38, 0x61, 0x2d, 0x5c,
	0x03, 0x3a, 0xc6, 0x88, 0x90, 0xdb, 0x8c, 0x8c,
	0x42, 0xf3, 0x51, 0x75, 0x43, 0xa0, 0x83, 0x93
])

def shift5(arr, num, fill_value = 0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def calc_laurent_h0(BT, L, N):
    # FIXME: shift by half a sample
    h0_len = (L + 1) * N
    h_0 = np.ones((2 * (L + 1) * N))
    q_t = calc_gmsk_phase_func(BT, L, N)
    c_t = np.cos(np.pi * q_t)
    c_t = np.insert(c_t, 0, np.flipud(c_t[1:]))
    cshift = np.zeros((2 * (L + 1) * N))
    cshift[:2 * L * N + 1] = c_t
    for l in range(L):
        h_0 = np.multiply(h_0, shift5(cshift, l * N))
        
    mid_idx = np.argmax(h_0)
    h_0_trunc = h_0[mid_idx - h0_len//2 : mid_idx + h0_len//2 + 1]

    return h_0_trunc

def G_func(sigma, x):
    return x * (1 - stats.norm.sf(x/sigma)) + sigma/np.sqrt(2 * np.pi) * np.exp(-x**2/2.0 / sigma**2)

def calc_gmsk_phase_func(BT, L, N):
    T = 1
    t = np.linspace(-L/float(2), L/float(2), L * N + 1)
    sigma = np.sqrt(np.log(2)) / (np.pi * BT * 2.0)
    f1 = G_func(sigma, t/T + 0.5)
    f2 = G_func(sigma, t/T - 0.5)
    return 0.5 * (f1 - f2)
    
def modulate_1p_alt2(h0, N):
    frame_bits = np.empty(0, dtype = np.uint8)
    for sp in sync_pattern:
        for b in range(8):
            frame_bits = np.append(frame_bits, (sp >> b) & 0x01)

    frame_bits_pre = np.empty_like(frame_bits)
    v = 0
    ocnt = 0
    for ii in np.arange(0, len(frame_bits), 2):
        frame_bits[ii] = 1 if frame_bits[ii] == 0 else 0
    for a in frame_bits:
        v = v ^ a
        frame_bits_pre[ocnt] = v
        ocnt += 1

        if(ocnt == 2 * 88):
            print(v)

    frame_bits_pre = 2 * frame_bits_pre - 1
    b_even = frame_bits_pre[::2]
    b_odd  = -frame_bits_pre[1::2]

    c = np.zeros( (2 * N - 1, len(b_even)), int)
    be_us = np.squeeze(np.vstack((b_even, c)).T.reshape(1, -1))
    bo_us = np.squeeze(np.vstack((b_odd, c)).T.reshape(1, -1))

    be_us = np.append(be_us, np.zeros((N,), int))
    bo_us = np.insert(bo_us, 0, np.zeros((N,), int))
    
    mod_e = signal.lfilter(h0, 1, be_us)
    mod_o = signal.lfilter(h0, 1, bo_us)
    return mod_e + 1j * mod_o

if __name__ == '__main__':
    oversamp = 20
    h0 = calc_laurent_h0(0.5, 2, oversamp)
    sync_iq = modulate_1p_alt2(h0, oversamp)
    plt.figure()
    plt.plot(np.unwrap(np.angle(sync_iq)))
    plt.show()