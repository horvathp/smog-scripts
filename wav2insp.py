import scipy.io.wavfile
import sys

filename = sys.argv[1]
fs1, y1 = scipy.io.wavfile.read(filename)
assert y1.dtype.name == 'int16'

stereo_to_complex = lambda a: a[0] + a[1] * 1j
y1 = y1/65536.
y1 = stereo_to_complex(y1.T).astype('complex64')
filename = filename.replace('wav', 'cf32')
y1.tofile(str(fs1) + '.' + filename)