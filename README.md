Comparing various launch configs for CUDA based **vector element sum**
(**in-place**).

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. This is an in-place sum, meaning the single sum values is
calculated entirely by the GPU. This is done using either 2 (if `grid_limit`
is `1024`) or 3 kernel calls (otherwise). A `block_size` of `128`
(decent choice for sum) is used for *2nd* kernel, if there are 3 kernels.
Sum here represents any reduction operation that processes several values
to a single value. Using a `grid_limit` of **1024** and a `block_size` of
**128** could be a decent choice (for float). Theere doesn't seem to be a
good enough advantage of doing multiple reads per sum-loop.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] can be generated later, if needed, also todo [sheets]. For
related experiments, see [branches]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# Elements 1e+09
# [01444.221 ms] [21.300482] sumSeq
# [00012.183 ms] [21.300482] sumCuda<<<1024, 32>>> [mode=0]
# [00015.230 ms] [21.300482] sumCuda<<<1024, 32>>> [mode=1]
# [00012.050 ms] [21.300482] sumCuda<<<1024, 32>>> [mode=2]
# [00010.874 ms] [21.300482] sumCuda<<<1024, 64>>> [mode=0]
# [00011.919 ms] [21.300482] sumCuda<<<1024, 64>>> [mode=1]
# [00010.756 ms] [21.300482] sumCuda<<<1024, 64>>> [mode=2]
# [00009.945 ms] [21.300482] sumCuda<<<1024, 128>>> [mode=0]
# [00010.899 ms] [21.300482] sumCuda<<<1024, 128>>> [mode=1]
# [00010.186 ms] [21.300482] sumCuda<<<1024, 128>>> [mode=2]
# [00008.997 ms] [21.300482] sumCuda<<<1024, 256>>> [mode=0]
# [00008.985 ms] [21.300482] sumCuda<<<1024, 256>>> [mode=1]
# [00008.980 ms] [21.300482] sumCuda<<<1024, 256>>> [mode=2]
# [00009.228 ms] [21.300482] sumCuda<<<1024, 512>>> [mode=0]
# [00009.201 ms] [21.300482] sumCuda<<<1024, 512>>> [mode=1]
# [00009.260 ms] [21.300482] sumCuda<<<1024, 512>>> [mode=2]
# [00008.999 ms] [21.300482] sumCuda<<<1024, 1024>>> [mode=0]
# [00008.971 ms] [21.300482] sumCuda<<<1024, 1024>>> [mode=1]
# [00008.995 ms] [21.300482] sumCuda<<<1024, 1024>>> [mode=2]
# [00010.343 ms] [21.300482] sumCuda<<<2048, 32>>> [mode=0]
# [00012.137 ms] [21.300482] sumCuda<<<2048, 32>>> [mode=1]
# [00010.256 ms] [21.300482] sumCuda<<<2048, 32>>> [mode=2]
# [00009.722 ms] [21.300482] sumCuda<<<2048, 64>>> [mode=0]
# [00010.257 ms] [21.300482] sumCuda<<<2048, 64>>> [mode=1]
# [00009.667 ms] [21.300482] sumCuda<<<2048, 64>>> [mode=2]
# [00009.587 ms] [21.300482] sumCuda<<<2048, 128>>> [mode=0]
# [00009.670 ms] [21.300482] sumCuda<<<2048, 128>>> [mode=1]
# [00009.444 ms] [21.300482] sumCuda<<<2048, 128>>> [mode=2]
# [00009.195 ms] [21.300482] sumCuda<<<2048, 256>>> [mode=0]
# [00009.212 ms] [21.300482] sumCuda<<<2048, 256>>> [mode=1]
# [00009.216 ms] [21.300482] sumCuda<<<2048, 256>>> [mode=2]
# [00008.976 ms] [21.300482] sumCuda<<<2048, 512>>> [mode=0]
# [00008.976 ms] [21.300482] sumCuda<<<2048, 512>>> [mode=1]
# [00008.998 ms] [21.300482] sumCuda<<<2048, 512>>> [mode=2]
# [00008.953 ms] [21.300482] sumCuda<<<2048, 1024>>> [mode=0]
# [00008.964 ms] [21.300482] sumCuda<<<2048, 1024>>> [mode=1]
# [00008.952 ms] [21.300482] sumCuda<<<2048, 1024>>> [mode=2]
# [00009.431 ms] [21.300482] sumCuda<<<4096, 32>>> [mode=0]
# [00010.686 ms] [21.300482] sumCuda<<<4096, 32>>> [mode=1]
# [00009.394 ms] [21.300482] sumCuda<<<4096, 32>>> [mode=2]
# [00009.139 ms] [21.300482] sumCuda<<<4096, 64>>> [mode=0]
# [00009.450 ms] [21.300482] sumCuda<<<4096, 64>>> [mode=1]
# [00009.125 ms] [21.300482] sumCuda<<<4096, 64>>> [mode=2]
# [00009.080 ms] [21.300482] sumCuda<<<4096, 128>>> [mode=0]
# [00009.134 ms] [21.300482] sumCuda<<<4096, 128>>> [mode=1]
# [00009.090 ms] [21.300482] sumCuda<<<4096, 128>>> [mode=2]
# [00008.972 ms] [21.300482] sumCuda<<<4096, 256>>> [mode=0]
# [00008.974 ms] [21.300482] sumCuda<<<4096, 256>>> [mode=1]
# [00008.967 ms] [21.300482] sumCuda<<<4096, 256>>> [mode=2]
# [00008.954 ms] [21.300482] sumCuda<<<4096, 512>>> [mode=0]
# [00008.977 ms] [21.300482] sumCuda<<<4096, 512>>> [mode=1]
# [00008.953 ms] [21.300482] sumCuda<<<4096, 512>>> [mode=2]
# [00008.972 ms] [21.300482] sumCuda<<<4096, 1024>>> [mode=0]
# [00008.956 ms] [21.300482] sumCuda<<<4096, 1024>>> [mode=1]
# [00008.951 ms] [21.300482] sumCuda<<<4096, 1024>>> [mode=2]
# [00009.093 ms] [21.300482] sumCuda<<<8192, 32>>> [mode=0]
# [00010.215 ms] [21.300482] sumCuda<<<8192, 32>>> [mode=1]
# [00009.109 ms] [21.300482] sumCuda<<<8192, 32>>> [mode=2]
# [00008.977 ms] [21.300482] sumCuda<<<8192, 64>>> [mode=0]
# [00009.135 ms] [21.300482] sumCuda<<<8192, 64>>> [mode=1]
# [00008.978 ms] [21.300482] sumCuda<<<8192, 64>>> [mode=2]
# [00009.020 ms] [21.300482] sumCuda<<<8192, 128>>> [mode=0]
# [00008.979 ms] [21.300482] sumCuda<<<8192, 128>>> [mode=1]
# [00009.003 ms] [21.300482] sumCuda<<<8192, 128>>> [mode=2]
# [00008.950 ms] [21.300482] sumCuda<<<8192, 256>>> [mode=0]
# [00008.963 ms] [21.300482] sumCuda<<<8192, 256>>> [mode=1]
# [00008.951 ms] [21.300482] sumCuda<<<8192, 256>>> [mode=2]
# [00008.971 ms] [21.300482] sumCuda<<<8192, 512>>> [mode=0]
# [00008.955 ms] [21.300482] sumCuda<<<8192, 512>>> [mode=1]
# [00008.943 ms] [21.300482] sumCuda<<<8192, 512>>> [mode=2]
# [00008.948 ms] [21.300482] sumCuda<<<8192, 1024>>> [mode=0]
# [00008.952 ms] [21.300482] sumCuda<<<8192, 1024>>> [mode=1]
# [00008.950 ms] [21.300482] sumCuda<<<8192, 1024>>> [mode=2]
# [00009.162 ms] [21.300482] sumCuda<<<16384, 32>>> [mode=0]
# [00010.113 ms] [21.300482] sumCuda<<<16384, 32>>> [mode=1]
# [00009.163 ms] [21.300482] sumCuda<<<16384, 32>>> [mode=2]
# [00009.045 ms] [21.300482] sumCuda<<<16384, 64>>> [mode=0]
# [00009.178 ms] [21.300482] sumCuda<<<16384, 64>>> [mode=1]
# [00009.026 ms] [21.300482] sumCuda<<<16384, 64>>> [mode=2]
# [00009.074 ms] [21.300482] sumCuda<<<16384, 128>>> [mode=0]
# [00009.061 ms] [21.300482] sumCuda<<<16384, 128>>> [mode=1]
# [00009.045 ms] [21.300482] sumCuda<<<16384, 128>>> [mode=2]
# [00008.960 ms] [21.300482] sumCuda<<<16384, 256>>> [mode=0]
# [00008.957 ms] [21.300482] sumCuda<<<16384, 256>>> [mode=1]
# [00008.976 ms] [21.300482] sumCuda<<<16384, 256>>> [mode=2]
# [00008.947 ms] [21.300482] sumCuda<<<16384, 512>>> [mode=0]
# [00008.954 ms] [21.300482] sumCuda<<<16384, 512>>> [mode=1]
# [00008.949 ms] [21.300482] sumCuda<<<16384, 512>>> [mode=2]
# [00008.944 ms] [21.300482] sumCuda<<<16384, 1024>>> [mode=0]
# [00008.950 ms] [21.300482] sumCuda<<<16384, 1024>>> [mode=1]
# [00008.947 ms] [21.300482] sumCuda<<<16384, 1024>>> [mode=2]
# [00009.061 ms] [21.300482] sumCuda<<<32768, 32>>> [mode=0]
# [00009.996 ms] [21.300482] sumCuda<<<32768, 32>>> [mode=1]
# [00009.071 ms] [21.300482] sumCuda<<<32768, 32>>> [mode=2]
# [00008.993 ms] [21.300482] sumCuda<<<32768, 64>>> [mode=0]
# [00009.135 ms] [21.300482] sumCuda<<<32768, 64>>> [mode=1]
# [00008.985 ms] [21.300482] sumCuda<<<32768, 64>>> [mode=2]
# [00008.986 ms] [21.300482] sumCuda<<<32768, 128>>> [mode=0]
# [00008.991 ms] [21.300482] sumCuda<<<32768, 128>>> [mode=1]
# [00008.989 ms] [21.300482] sumCuda<<<32768, 128>>> [mode=2]
# [00008.949 ms] [21.300482] sumCuda<<<32768, 256>>> [mode=0]
# [00008.954 ms] [21.300482] sumCuda<<<32768, 256>>> [mode=1]
# [00008.953 ms] [21.300482] sumCuda<<<32768, 256>>> [mode=2]
# [00008.945 ms] [21.300482] sumCuda<<<32768, 512>>> [mode=0]
# [00008.953 ms] [21.300482] sumCuda<<<32768, 512>>> [mode=1]
# [00008.947 ms] [21.300482] sumCuda<<<32768, 512>>> [mode=2]
# [00008.946 ms] [21.300482] sumCuda<<<32768, 1024>>> [mode=0]
# [00008.949 ms] [21.300482] sumCuda<<<32768, 1024>>> [mode=1]
# [00008.947 ms] [21.300482] sumCuda<<<32768, 1024>>> [mode=2]
```

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)

<br>
<br>

[![](https://i.imgur.com/SrEEKD5.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[branches]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch/branches
[charts]: https://wolfram77.github.io
[sheets]: https://wolfram77.github.io
