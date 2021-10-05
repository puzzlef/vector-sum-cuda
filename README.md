Comparing various *launch configs* for CUDA based **vector element sum**
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
# # Elements 1e+07
# [00014.500 ms] [15.403683] sumSeq
# [00000.080 ms] [16.695312] sumCuda<<<1024, 32>>>
# [00000.074 ms] [16.695312] sumCuda<<<1024, 64>>>
# [00000.074 ms] [16.695312] sumCuda<<<1024, 128>>>
# [00000.076 ms] [16.695312] sumCuda<<<1024, 256>>>
# [00000.075 ms] [16.695312] sumCuda<<<1024, 512>>>
# [00000.078 ms] [16.695312] sumCuda<<<1024, 1024>>>
# [00000.090 ms] [16.695311] sumCuda<<<2048, 32>>>
# [00000.083 ms] [16.695311] sumCuda<<<2048, 64>>>
# [00000.076 ms] [16.695311] sumCuda<<<2048, 128>>>
# [00000.077 ms] [16.695312] sumCuda<<<2048, 256>>>
# [00000.078 ms] [16.695311] sumCuda<<<2048, 512>>>
# [00000.090 ms] [16.695311] sumCuda<<<2048, 1024>>>
# [00000.084 ms] [16.695311] sumCuda<<<4096, 32>>>
# [00000.077 ms] [16.695311] sumCuda<<<4096, 64>>>
# [00000.075 ms] [16.695311] sumCuda<<<4096, 128>>>
# [00000.076 ms] [16.695312] sumCuda<<<4096, 256>>>
# [00000.079 ms] [16.695311] sumCuda<<<4096, 512>>>
# [00000.134 ms] [16.695311] sumCuda<<<4096, 1024>>>
# [00000.081 ms] [16.695311] sumCuda<<<8192, 32>>>
# [00000.075 ms] [16.695311] sumCuda<<<8192, 64>>>
# [00000.078 ms] [16.695312] sumCuda<<<8192, 128>>>
# [00000.079 ms] [16.695312] sumCuda<<<8192, 256>>>
# [00000.118 ms] [16.695311] sumCuda<<<8192, 512>>>
# [00000.214 ms] [16.695311] sumCuda<<<8192, 1024>>>
# [00000.084 ms] [16.695311] sumCuda<<<16384, 32>>>
# [00000.080 ms] [16.695311] sumCuda<<<16384, 64>>>
# [00000.080 ms] [16.695311] sumCuda<<<16384, 128>>>
# [00000.110 ms] [16.695311] sumCuda<<<16384, 256>>>
# [00000.176 ms] [16.695312] sumCuda<<<16384, 512>>>
# [00000.243 ms] [16.272846] sumCuda<<<16384, 1024>>>
# [00000.087 ms] [16.695311] sumCuda<<<32768, 32>>>
# [00000.082 ms] [16.695311] sumCuda<<<32768, 64>>>
# [00000.096 ms] [16.695311] sumCuda<<<32768, 128>>>
# [00000.150 ms] [16.695311] sumCuda<<<32768, 256>>>
# [00000.185 ms] [16.168745] sumCuda<<<32768, 512>>>
# [00000.224 ms] [16.272846] sumCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/CWySswQ.gif)][sheetp]
[![](https://i.imgur.com/o3mYdbR.gif)][sheetp]
[![](https://i.imgur.com/jGqYBwP.gif)][sheetp]
[![](https://i.imgur.com/ktH8eSd.gif)][sheetp]

[![](https://i.imgur.com/5ptA5mP.png)][sheetp]
[![](https://i.imgur.com/H5fsuAu.png)][sheetp]
[![](https://i.imgur.com/me2NSUv.png)][sheetp]
[![](https://i.imgur.com/Acuh8hF.png)][sheetp]

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
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vRR3VIK58QcfE3fDl2EMhg8TKvZQOq4QONU3WkcDZNihlzG82gtROy4QknkcN5xHlWyraIEtteS4YI2/pubhtml
