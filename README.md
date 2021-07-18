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
to a single value.

A number of possible optimizations including *multiple reads per loop*
*iteration*, *loop unrolled reduce*, and *atomic adds* provided no benefit
(see [branches]). A simple **one read per loop iteration** and **standard**
**reduce loop** (minimizing warp divergence) is both **shorter** and **works**
**best**. For **float**, a `grid_limit` of **1024** and a `block_size` of
**128** is a decent choice. For **double**, a `grid_limit` of **1024** and a
`block_size` of **256** is a decent choice.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. For
related experiments, see [branches]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+09
# [01445.962 ms] [21.300482] sumSeq
# [00012.172 ms] [21.300482] sumCuda<<<1024, 32>>>
# [00010.875 ms] [21.300482] sumCuda<<<1024, 64>>>
# [00009.824 ms] [21.300482] sumCuda<<<1024, 128>>>
# [00008.983 ms] [21.300482] sumCuda<<<1024, 256>>>
# [00009.236 ms] [21.300482] sumCuda<<<1024, 512>>>
# [00008.996 ms] [21.300482] sumCuda<<<1024, 1024>>>
# [00010.344 ms] [21.300482] sumCuda<<<2048, 32>>>
# [00009.722 ms] [21.300482] sumCuda<<<2048, 64>>>
# [00009.587 ms] [21.300482] sumCuda<<<2048, 128>>>
# [00009.194 ms] [21.300482] sumCuda<<<2048, 256>>>
# [00008.976 ms] [21.300482] sumCuda<<<2048, 512>>>
# [00008.954 ms] [21.300482] sumCuda<<<2048, 1024>>>
# [00009.431 ms] [21.300482] sumCuda<<<4096, 32>>>
# [00009.126 ms] [21.300482] sumCuda<<<4096, 64>>>
# [00009.108 ms] [21.300482] sumCuda<<<4096, 128>>>
# [00008.978 ms] [21.300482] sumCuda<<<4096, 256>>>
# [00008.950 ms] [21.300482] sumCuda<<<4096, 512>>>
# [00008.968 ms] [21.300482] sumCuda<<<4096, 1024>>>
# [00009.098 ms] [21.300482] sumCuda<<<8192, 32>>>
# [00008.976 ms] [21.300482] sumCuda<<<8192, 64>>>
# [00009.019 ms] [21.300482] sumCuda<<<8192, 128>>>
# [00008.948 ms] [21.300482] sumCuda<<<8192, 256>>>
# [00008.964 ms] [21.300482] sumCuda<<<8192, 512>>>
# [00008.950 ms] [21.300482] sumCuda<<<8192, 1024>>>
# [00009.158 ms] [21.300482] sumCuda<<<16384, 32>>>
# [00009.044 ms] [21.300482] sumCuda<<<16384, 64>>>
# [00009.077 ms] [21.300482] sumCuda<<<16384, 128>>>
# [00008.956 ms] [21.300482] sumCuda<<<16384, 256>>>
# [00008.948 ms] [21.300482] sumCuda<<<16384, 512>>>
# [00008.939 ms] [21.300482] sumCuda<<<16384, 1024>>>
# [00009.052 ms] [21.300482] sumCuda<<<32768, 32>>>
# [00008.989 ms] [21.300482] sumCuda<<<32768, 64>>>
# [00008.991 ms] [21.300482] sumCuda<<<32768, 128>>>
# [00008.949 ms] [21.300482] sumCuda<<<32768, 256>>>
# [00008.948 ms] [21.300482] sumCuda<<<32768, 512>>>
# [00008.947 ms] [21.300482] sumCuda<<<32768, 1024>>>
```

[![](https://i.imgur.com/CWySswQ.gif)][sheets]
[![](https://i.imgur.com/o3mYdbR.gif)][sheets]
[![](https://i.imgur.com/jGqYBwP.gif)][sheets]
[![](https://i.imgur.com/ktH8eSd.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)
- [Optimizing Parallel Reduction in CUDA :: Mark Harris](https://www.slideshare.net/SubhajitSahu/optimizing-parallel-reduction-in-cuda-notes)

<br>
<br>

[![](https://i.imgur.com/s6FklYl.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[branches]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch/branches
[charts]: https://photos.app.goo.gl/795Rcbqa14srjoZBA
[sheets]: https://docs.google.com/spreadsheets/d/1pgIn6dcrKtVv0SoaJeQwTe1CzHRKuoUOXjn5_KJqrA8/edit?usp=sharing
