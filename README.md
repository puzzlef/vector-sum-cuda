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
any benefit of doing loop unrolled reduce (with next-pow-2 for block size).

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
# # Elements 1e+09
# [01447.563 ms] [21.300482] sumSeq
# [00012.134 ms] [21.300482] sumCuda<<<1024, 32>>>
# [00010.856 ms] [21.300482] sumCuda<<<1024, 64>>>
# [00009.962 ms] [21.300482] sumCuda<<<1024, 128>>>
# [00008.995 ms] [21.300482] sumCuda<<<1024, 256>>>
# [00009.214 ms] [21.300482] sumCuda<<<1024, 512>>>
# [00008.973 ms] [21.300482] sumCuda<<<1024, 1024>>>
# [00010.323 ms] [21.300482] sumCuda<<<2048, 32>>>
# [00009.718 ms] [21.300482] sumCuda<<<2048, 64>>>
# [00009.593 ms] [21.300482] sumCuda<<<2048, 128>>>
# [00009.201 ms] [21.300482] sumCuda<<<2048, 256>>>
# [00008.986 ms] [21.300482] sumCuda<<<2048, 512>>>
# [00008.955 ms] [21.300482] sumCuda<<<2048, 1024>>>
# [00009.420 ms] [21.300482] sumCuda<<<4096, 32>>>
# [00009.135 ms] [21.300482] sumCuda<<<4096, 64>>>
# [00009.107 ms] [21.300482] sumCuda<<<4096, 128>>>
# [00008.972 ms] [21.300482] sumCuda<<<4096, 256>>>
# [00008.952 ms] [21.300482] sumCuda<<<4096, 512>>>
# [00008.982 ms] [21.300482] sumCuda<<<4096, 1024>>>
# [00009.101 ms] [21.300482] sumCuda<<<8192, 32>>>
# [00008.977 ms] [21.300482] sumCuda<<<8192, 64>>>
# [00009.017 ms] [21.300482] sumCuda<<<8192, 128>>>
# [00008.949 ms] [21.300482] sumCuda<<<8192, 256>>>
# [00008.973 ms] [21.300482] sumCuda<<<8192, 512>>>
# [00008.946 ms] [21.300482] sumCuda<<<8192, 1024>>>
# [00009.160 ms] [21.300482] sumCuda<<<16384, 32>>>
# [00009.041 ms] [21.300482] sumCuda<<<16384, 64>>>
# [00009.083 ms] [21.300482] sumCuda<<<16384, 128>>>
# [00008.956 ms] [21.300482] sumCuda<<<16384, 256>>>
# [00008.945 ms] [21.300482] sumCuda<<<16384, 512>>>
# [00008.937 ms] [21.300482] sumCuda<<<16384, 1024>>>
# [00009.062 ms] [21.300482] sumCuda<<<32768, 32>>>
# [00008.990 ms] [21.300482] sumCuda<<<32768, 64>>>
# [00008.992 ms] [21.300482] sumCuda<<<32768, 128>>>
# [00008.946 ms] [21.300482] sumCuda<<<32768, 256>>>
# [00008.946 ms] [21.300482] sumCuda<<<32768, 512>>>
# [00008.945 ms] [21.300482] sumCuda<<<32768, 1024>>>
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
