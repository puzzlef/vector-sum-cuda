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
**128** could be a decent choice (for float). There doesn't seem to be
any benefit of doing multiple reads per sum-loop in block steps (using a
template and prev-pow-2 for block size).

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
# [01444.784 ms] [21.300482] sumSeq
# [00012.184 ms] [21.300482] sumCuda<<<1024, 32>>> [mode=0]
# [00015.147 ms] [21.300482] sumCuda<<<1024, 32>>> [mode=1]
# [00011.963 ms] [21.300482] sumCuda<<<1024, 32>>> [mode=2]
# [00010.866 ms] [21.300482] sumCuda<<<1024, 64>>> [mode=0]
# [00011.900 ms] [21.300482] sumCuda<<<1024, 64>>> [mode=1]
# [00010.707 ms] [21.300482] sumCuda<<<1024, 64>>> [mode=2]
# [00010.079 ms] [21.300482] sumCuda<<<1024, 128>>> [mode=0]
# [00010.776 ms] [21.300482] sumCuda<<<1024, 128>>> [mode=1]
# [00010.339 ms] [21.300482] sumCuda<<<1024, 128>>> [mode=2]
# [00008.980 ms] [21.300482] sumCuda<<<1024, 256>>> [mode=0]
# [00008.996 ms] [21.300482] sumCuda<<<1024, 256>>> [mode=1]
# [00009.002 ms] [21.300482] sumCuda<<<1024, 256>>> [mode=2]
# [00009.240 ms] [21.300482] sumCuda<<<1024, 512>>> [mode=0]
# [00009.184 ms] [21.300482] sumCuda<<<1024, 512>>> [mode=1]
# [00009.325 ms] [21.300482] sumCuda<<<1024, 512>>> [mode=2]
# [00008.998 ms] [21.300482] sumCuda<<<1024, 1024>>> [mode=0]
# [00009.023 ms] [21.300482] sumCuda<<<1024, 1024>>> [mode=1]
# [00009.055 ms] [21.300482] sumCuda<<<1024, 1024>>> [mode=2]
# [00010.345 ms] [21.300482] sumCuda<<<2048, 32>>> [mode=0]
# [00012.222 ms] [21.300482] sumCuda<<<2048, 32>>> [mode=1]
# [00010.275 ms] [21.300482] sumCuda<<<2048, 32>>> [mode=2]
# [00009.723 ms] [21.300482] sumCuda<<<2048, 64>>> [mode=0]
# [00010.222 ms] [21.300482] sumCuda<<<2048, 64>>> [mode=1]
# [00009.606 ms] [21.300482] sumCuda<<<2048, 64>>> [mode=2]
# [00009.589 ms] [21.300482] sumCuda<<<2048, 128>>> [mode=0]
# [00009.620 ms] [21.300482] sumCuda<<<2048, 128>>> [mode=1]
# [00009.489 ms] [21.300482] sumCuda<<<2048, 128>>> [mode=2]
# [00009.200 ms] [21.300482] sumCuda<<<2048, 256>>> [mode=0]
# [00009.176 ms] [21.300482] sumCuda<<<2048, 256>>> [mode=1]
# [00009.305 ms] [21.300482] sumCuda<<<2048, 256>>> [mode=2]
# [00008.973 ms] [21.300482] sumCuda<<<2048, 512>>> [mode=0]
# [00009.006 ms] [21.300482] sumCuda<<<2048, 512>>> [mode=1]
# [00009.046 ms] [21.300482] sumCuda<<<2048, 512>>> [mode=2]
# [00008.949 ms] [21.300482] sumCuda<<<2048, 1024>>> [mode=0]
# [00008.973 ms] [21.300482] sumCuda<<<2048, 1024>>> [mode=1]
# [00008.965 ms] [21.300482] sumCuda<<<2048, 1024>>> [mode=2]
# [00009.427 ms] [21.300482] sumCuda<<<4096, 32>>> [mode=0]
# [00010.773 ms] [21.300482] sumCuda<<<4096, 32>>> [mode=1]
# [00009.437 ms] [21.300482] sumCuda<<<4096, 32>>> [mode=2]
# [00009.131 ms] [21.300482] sumCuda<<<4096, 64>>> [mode=0]
# [00009.419 ms] [21.300482] sumCuda<<<4096, 64>>> [mode=1]
# [00009.106 ms] [21.300482] sumCuda<<<4096, 64>>> [mode=2]
# [00009.099 ms] [21.300482] sumCuda<<<4096, 128>>> [mode=0]
# [00009.109 ms] [21.300482] sumCuda<<<4096, 128>>> [mode=1]
# [00009.123 ms] [21.300482] sumCuda<<<4096, 128>>> [mode=2]
# [00008.974 ms] [21.300482] sumCuda<<<4096, 256>>> [mode=0]
# [00009.005 ms] [21.300482] sumCuda<<<4096, 256>>> [mode=1]
# [00009.023 ms] [21.300482] sumCuda<<<4096, 256>>> [mode=2]
# [00008.951 ms] [21.300482] sumCuda<<<4096, 512>>> [mode=0]
# [00008.964 ms] [21.300482] sumCuda<<<4096, 512>>> [mode=1]
# [00008.969 ms] [21.300482] sumCuda<<<4096, 512>>> [mode=2]
# [00008.970 ms] [21.300482] sumCuda<<<4096, 1024>>> [mode=0]
# [00008.963 ms] [21.300482] sumCuda<<<4096, 1024>>> [mode=1]
# [00008.968 ms] [21.300482] sumCuda<<<4096, 1024>>> [mode=2]
# [00009.092 ms] [21.300482] sumCuda<<<8192, 32>>> [mode=0]
# [00010.071 ms] [21.300482] sumCuda<<<8192, 32>>> [mode=1]
# [00009.087 ms] [21.300482] sumCuda<<<8192, 32>>> [mode=2]
# [00008.974 ms] [21.300482] sumCuda<<<8192, 64>>> [mode=0]
# [00009.099 ms] [21.300482] sumCuda<<<8192, 64>>> [mode=1]
# [00008.968 ms] [21.300482] sumCuda<<<8192, 64>>> [mode=2]
# [00009.014 ms] [21.300482] sumCuda<<<8192, 128>>> [mode=0]
# [00008.988 ms] [21.300482] sumCuda<<<8192, 128>>> [mode=1]
# [00008.999 ms] [21.300482] sumCuda<<<8192, 128>>> [mode=2]
# [00008.948 ms] [21.300482] sumCuda<<<8192, 256>>> [mode=0]
# [00008.962 ms] [21.300482] sumCuda<<<8192, 256>>> [mode=1]
# [00008.957 ms] [21.300482] sumCuda<<<8192, 256>>> [mode=2]
# [00008.959 ms] [21.300482] sumCuda<<<8192, 512>>> [mode=0]
# [00008.957 ms] [21.300482] sumCuda<<<8192, 512>>> [mode=1]
# [00008.952 ms] [21.300482] sumCuda<<<8192, 512>>> [mode=2]
# [00008.948 ms] [21.300482] sumCuda<<<8192, 1024>>> [mode=0]
# [00008.965 ms] [21.300482] sumCuda<<<8192, 1024>>> [mode=1]
# [00008.952 ms] [21.300482] sumCuda<<<8192, 1024>>> [mode=2]
# [00009.161 ms] [21.300482] sumCuda<<<16384, 32>>> [mode=0]
# [00010.106 ms] [21.300482] sumCuda<<<16384, 32>>> [mode=1]
# [00009.178 ms] [21.300482] sumCuda<<<16384, 32>>> [mode=2]
# [00009.045 ms] [21.300482] sumCuda<<<16384, 64>>> [mode=0]
# [00009.203 ms] [21.300482] sumCuda<<<16384, 64>>> [mode=1]
# [00009.041 ms] [21.300482] sumCuda<<<16384, 64>>> [mode=2]
# [00009.074 ms] [21.300482] sumCuda<<<16384, 128>>> [mode=0]
# [00009.033 ms] [21.300482] sumCuda<<<16384, 128>>> [mode=1]
# [00009.007 ms] [21.300482] sumCuda<<<16384, 128>>> [mode=2]
# [00008.958 ms] [21.300482] sumCuda<<<16384, 256>>> [mode=0]
# [00008.959 ms] [21.300482] sumCuda<<<16384, 256>>> [mode=1]
# [00008.957 ms] [21.300482] sumCuda<<<16384, 256>>> [mode=2]
# [00008.946 ms] [21.300482] sumCuda<<<16384, 512>>> [mode=0]
# [00008.958 ms] [21.300482] sumCuda<<<16384, 512>>> [mode=1]
# [00008.950 ms] [21.300482] sumCuda<<<16384, 512>>> [mode=2]
# [00008.947 ms] [21.300482] sumCuda<<<16384, 1024>>> [mode=0]
# [00008.947 ms] [21.300482] sumCuda<<<16384, 1024>>> [mode=1]
# [00008.947 ms] [21.300482] sumCuda<<<16384, 1024>>> [mode=2]
# [00009.053 ms] [21.300482] sumCuda<<<32768, 32>>> [mode=0]
# [00010.005 ms] [21.300482] sumCuda<<<32768, 32>>> [mode=1]
# [00009.080 ms] [21.300482] sumCuda<<<32768, 32>>> [mode=2]
# [00008.990 ms] [21.300482] sumCuda<<<32768, 64>>> [mode=0]
# [00009.094 ms] [21.300482] sumCuda<<<32768, 64>>> [mode=1]
# [00008.982 ms] [21.300482] sumCuda<<<32768, 64>>> [mode=2]
# [00008.991 ms] [21.300482] sumCuda<<<32768, 128>>> [mode=0]
# [00008.985 ms] [21.300482] sumCuda<<<32768, 128>>> [mode=1]
# [00009.012 ms] [21.300482] sumCuda<<<32768, 128>>> [mode=2]
# [00008.949 ms] [21.300482] sumCuda<<<32768, 256>>> [mode=0]
# [00008.966 ms] [21.300482] sumCuda<<<32768, 256>>> [mode=1]
# [00008.952 ms] [21.300482] sumCuda<<<32768, 256>>> [mode=2]
# [00008.946 ms] [21.300482] sumCuda<<<32768, 512>>> [mode=0]
# [00008.948 ms] [21.300482] sumCuda<<<32768, 512>>> [mode=1]
# [00008.948 ms] [21.300482] sumCuda<<<32768, 512>>> [mode=2]
# [00008.946 ms] [21.300482] sumCuda<<<32768, 1024>>> [mode=0]
# [00008.947 ms] [21.300482] sumCuda<<<32768, 1024>>> [mode=1]
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
