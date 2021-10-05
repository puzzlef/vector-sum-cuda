Comparing various *launch configs* for CUDA based **vector element sum**
(**in-place**).

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. This is an in-place sum, meaning the single sum values is
calculated entirely by the GPU. This is done using 2 kernel calls. Sum here
represents any reduction operation that processes several values to a
single value.

A number of possible optimizations including *multiple reads per loop*
*iteration*, *loop unrolled reduce*, *atomic adds*, and *multiple kernels*
provided no benefit (see [branches]). A simple **one read per loop iteration**
and **standard reduce loop** (minimizing warp divergence) is both **shorter** and
**works best**. For **float**, a **grid_limit** of `1024` and a **block_size** of
`128` is a decent choice. For **double**, a **grid_limit** of `1024` and a
**block_size** of `256` is a decent choice. Interestingly, the *sequential sum*
suffers from **precision issue** when using the **float** datatype, while the
*CUDA based sum* does not (just like with [memcpy sum]).

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
# [00014.251 ms] [15.403683] sumSeq
# [00000.080 ms] [16.695312] sumCuda<<<1024, 32>>>
# [00000.074 ms] [16.695312] sumCuda<<<1024, 64>>>
# [00000.072 ms] [16.695312] sumCuda<<<1024, 128>>>
# [00000.075 ms] [16.695312] sumCuda<<<1024, 256>>>
# [00000.075 ms] [16.695312] sumCuda<<<1024, 512>>>
# [00000.078 ms] [16.695312] sumCuda<<<1024, 1024>>>
# [00000.088 ms] [16.695312] sumCuda<<<2048, 32>>>
# [00000.081 ms] [16.695312] sumCuda<<<2048, 64>>>
# [00000.075 ms] [16.695312] sumCuda<<<2048, 128>>>
# [00000.075 ms] [16.695312] sumCuda<<<2048, 256>>>
# [00000.076 ms] [16.695312] sumCuda<<<2048, 512>>>
# [00000.086 ms] [16.695311] sumCuda<<<2048, 1024>>>
# [00000.083 ms] [16.695311] sumCuda<<<4096, 32>>>
# [00000.076 ms] [16.695311] sumCuda<<<4096, 64>>>
# [00000.074 ms] [16.695312] sumCuda<<<4096, 128>>>
# [00000.074 ms] [16.695312] sumCuda<<<4096, 256>>>
# [00000.078 ms] [16.695312] sumCuda<<<4096, 512>>>
# [00000.132 ms] [16.695311] sumCuda<<<4096, 1024>>>
# [00000.079 ms] [16.695311] sumCuda<<<8192, 32>>>
# [00000.075 ms] [16.695311] sumCuda<<<8192, 64>>>
# [00000.075 ms] [16.695312] sumCuda<<<8192, 128>>>
# [00000.076 ms] [16.695312] sumCuda<<<8192, 256>>>
# [00000.115 ms] [16.695312] sumCuda<<<8192, 512>>>
# [00000.213 ms] [16.695312] sumCuda<<<8192, 1024>>>
# [00000.081 ms] [16.695312] sumCuda<<<16384, 32>>>
# [00000.077 ms] [16.695312] sumCuda<<<16384, 64>>>
# [00000.076 ms] [16.695312] sumCuda<<<16384, 128>>>
# [00000.107 ms] [16.695312] sumCuda<<<16384, 256>>>
# [00000.164 ms] [16.695312] sumCuda<<<16384, 512>>>
# [00000.222 ms] [16.695312] sumCuda<<<16384, 1024>>>
# [00000.087 ms] [16.695312] sumCuda<<<32768, 32>>>
# [00000.083 ms] [16.695312] sumCuda<<<32768, 64>>>
# [00000.098 ms] [16.695312] sumCuda<<<32768, 128>>>
# [00000.151 ms] [16.695312] sumCuda<<<32768, 256>>>
# [00000.185 ms] [16.695312] sumCuda<<<32768, 512>>>
# [00000.222 ms] [16.695312] sumCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/CWySswQ.gif)][sheetp]
[![](https://i.imgur.com/o3mYdbR.gif)][sheetp]
[![](https://i.imgur.com/jGqYBwP.gif)][sheetp]
[![](https://i.imgur.com/ktH8eSd.gif)][sheetp]

[![](https://i.imgur.com/3TQJasB.png)][sheetp]
[![](https://i.imgur.com/rz9xIMk.png)][sheetp]
[![](https://i.imgur.com/73MYASb.png)][sheetp]
[![](https://i.imgur.com/5PaS4kC.png)][sheetp]

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
[memcpy sum]: https://github.com/puzzlef/sum-cuda-memcpy-adjust-launch
[branches]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch/branches
[charts]: https://photos.app.goo.gl/795Rcbqa14srjoZBA
[sheets]: https://docs.google.com/spreadsheets/d/1pgIn6dcrKtVv0SoaJeQwTe1CzHRKuoUOXjn5_KJqrA8/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vRR3VIK58QcfE3fDl2EMhg8TKvZQOq4QONU3WkcDZNihlzG82gtROy4QknkcN5xHlWyraIEtteS4YI2/pubhtml
