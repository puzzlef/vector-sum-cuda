Comparing various launch configs for CUDA based **vector element sum**.

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. Sum here represents any reduction operation that processes
several values to a single value.

This sum uses *memcpy* to transfer partial results to CPU, where the final sum
is calculated. If the result can be used within GPU itself, it might be faster
to calculate complete sum [in-place] instead of transferring to CPU. For
**float**, a `grid_limit` of **1024** and a `block_size` of **128** is a
decent choice. For **double**, a `grid_limit` of **1024** and a `block_size`
of **256** is a decent choice.

All outputs are saved in [out](out/) and a small part of the output is listed
here. [Nsight Compute] profile results are saved in [prof](prof/). Some [charts]
are also included below, generated from [sheets]. This experiment was done with
guidance from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# Elements 1e+09
# [01445.014 ms] [21.300482] sumSeq
# [00012.166 ms] [21.300482] sumCuda<<<1024, 32>>>
# [00010.884 ms] [21.300482] sumCuda<<<1024, 64>>>
# [00009.966 ms] [21.300482] sumCuda<<<1024, 128>>>
# [00008.996 ms] [21.300482] sumCuda<<<1024, 256>>>
# [00009.232 ms] [21.300482] sumCuda<<<1024, 512>>>
# [00009.002 ms] [21.300482] sumCuda<<<1024, 1024>>>
# [00010.346 ms] [21.300482] sumCuda<<<2048, 32>>>
# [00009.734 ms] [21.300482] sumCuda<<<2048, 64>>>
# [00009.569 ms] [21.300482] sumCuda<<<2048, 128>>>
# [00009.211 ms] [21.300482] sumCuda<<<2048, 256>>>
# [00008.989 ms] [21.300482] sumCuda<<<2048, 512>>>
# [00008.962 ms] [21.300482] sumCuda<<<2048, 1024>>>
# [00009.452 ms] [21.300482] sumCuda<<<4096, 32>>>
# [00009.144 ms] [21.300482] sumCuda<<<4096, 64>>>
# [00009.112 ms] [21.300482] sumCuda<<<4096, 128>>>
# [00008.995 ms] [21.300482] sumCuda<<<4096, 256>>>
# [00008.971 ms] [21.300482] sumCuda<<<4096, 512>>>
# [00008.991 ms] [21.300482] sumCuda<<<4096, 1024>>>
# [00009.138 ms] [21.300482] sumCuda<<<8192, 32>>>
# [00009.013 ms] [21.300482] sumCuda<<<8192, 64>>>
# [00009.046 ms] [21.300482] sumCuda<<<8192, 128>>>
# [00008.979 ms] [21.300482] sumCuda<<<8192, 256>>>
# [00008.999 ms] [21.300482] sumCuda<<<8192, 512>>>
# [00008.982 ms] [21.300482] sumCuda<<<8192, 1024>>>
# [00009.221 ms] [21.300482] sumCuda<<<16384, 32>>>
# [00009.105 ms] [21.300482] sumCuda<<<16384, 64>>>
# [00009.129 ms] [21.300482] sumCuda<<<16384, 128>>>
# [00009.014 ms] [21.300482] sumCuda<<<16384, 256>>>
# [00009.007 ms] [21.300482] sumCuda<<<16384, 512>>>
# [00009.006 ms] [21.300482] sumCuda<<<16384, 1024>>>
# [00009.182 ms] [21.300482] sumCuda<<<32768, 32>>>
# [00009.109 ms] [21.300482] sumCuda<<<32768, 64>>>
# [00009.110 ms] [21.300482] sumCuda<<<32768, 128>>>
# [00009.063 ms] [21.300482] sumCuda<<<32768, 256>>>
# [00009.071 ms] [21.300482] sumCuda<<<32768, 512>>>
# [00009.060 ms] [21.300482] sumCuda<<<32768, 1024>>>
```

[![](https://i.imgur.com/EseR7Oa.gif)][sheets]
[![](https://i.imgur.com/8TCXC0q.gif)][sheets]
[![](https://i.imgur.com/AmQEavQ.gif)][sheets]
[![](https://i.imgur.com/U9YlVC4.gif)][sheets]
[![](https://i.imgur.com/ZLqcgXy.gif)][sheets]
[![](https://i.imgur.com/HOx7ppf.gif)][sheets]
[![](https://i.imgur.com/OU4iUc5.gif)][sheets]
[![](https://i.imgur.com/yDIzLIW.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)

<br>
<br>

[![](https://i.imgur.com/s6FklYl.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[Nsight Compute]: https://developer.nvidia.com/nsight-compute
[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
[charts]: https://photos.app.goo.gl/Jytw1qgSFPoTrL1FA
[sheets]: https://docs.google.com/spreadsheets/d/1jNNg43h19DUNwdwQVkOsLe5Dq-f_MfIbzaGG1pai4SU/edit?usp=sharing
