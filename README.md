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
any benefit of doing loop unrolled reduce (with prev-pow-2 for block size).

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
# [01443.514 ms] [21.300482] sumSeq
# [00012.136 ms] [21.300482] sumCuda<<<1024, 32>>>
# [00010.856 ms] [21.300482] sumCuda<<<1024, 64>>>
# [00010.127 ms] [21.300482] sumCuda<<<1024, 128>>>
# [00008.992 ms] [21.300482] sumCuda<<<1024, 256>>>
# [00009.231 ms] [21.300482] sumCuda<<<1024, 512>>>
# [00008.977 ms] [21.300482] sumCuda<<<1024, 1024>>>
# [00010.326 ms] [21.300482] sumCuda<<<2048, 32>>>
# [00009.717 ms] [21.300482] sumCuda<<<2048, 64>>>
# [00009.575 ms] [21.300482] sumCuda<<<2048, 128>>>
# [00009.188 ms] [21.300482] sumCuda<<<2048, 256>>>
# [00008.978 ms] [21.300482] sumCuda<<<2048, 512>>>
# [00008.954 ms] [21.300482] sumCuda<<<2048, 1024>>>
# [00009.421 ms] [21.300482] sumCuda<<<4096, 32>>>
# [00009.120 ms] [21.300482] sumCuda<<<4096, 64>>>
# [00009.105 ms] [21.300482] sumCuda<<<4096, 128>>>
# [00008.972 ms] [21.300482] sumCuda<<<4096, 256>>>
# [00008.949 ms] [21.300482] sumCuda<<<4096, 512>>>
# [00008.966 ms] [21.300482] sumCuda<<<4096, 1024>>>
# [00009.103 ms] [21.300482] sumCuda<<<8192, 32>>>
# [00008.977 ms] [21.300482] sumCuda<<<8192, 64>>>
# [00009.013 ms] [21.300482] sumCuda<<<8192, 128>>>
# [00008.948 ms] [21.300482] sumCuda<<<8192, 256>>>
# [00008.972 ms] [21.300482] sumCuda<<<8192, 512>>>
# [00008.947 ms] [21.300482] sumCuda<<<8192, 1024>>>
# [00009.156 ms] [21.300482] sumCuda<<<16384, 32>>>
# [00009.037 ms] [21.300482] sumCuda<<<16384, 64>>>
# [00009.079 ms] [21.300482] sumCuda<<<16384, 128>>>
# [00008.957 ms] [21.300482] sumCuda<<<16384, 256>>>
# [00008.944 ms] [21.300482] sumCuda<<<16384, 512>>>
# [00008.946 ms] [21.300482] sumCuda<<<16384, 1024>>>
# [00009.051 ms] [21.300482] sumCuda<<<32768, 32>>>
# [00008.995 ms] [21.300482] sumCuda<<<32768, 64>>>
# [00008.989 ms] [21.300482] sumCuda<<<32768, 128>>>
# [00008.947 ms] [21.300482] sumCuda<<<32768, 256>>>
# [00008.944 ms] [21.300482] sumCuda<<<32768, 512>>>
# [00008.944 ms] [21.300482] sumCuda<<<32768, 1024>>>
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
