Comparing various launch configs for CUDA based **vector element sum**.

A floating-point vector `x`, with number of **elements** from `1E+6` to
`1E+9` was summed up using CUDA (`Σx`). Each element count was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. Sum here represents any `reduce()` operation that processes
several values to a single value.

This sum uses *memcpy* to transfer partial results to CPU, where the final sum
is calculated. If the result can be used within GPU itself, it *might* be
faster to calculate complete sum [in-place] instead of transferring to CPU.
Results indicate that a **grid_limit** of `1024` and a **block_size** of
`128/256` is suitable for **float** datatype, and a **grid_limit** of `1024`
and a **block_size** of `256` is suitable for **double** datatype. Thus, using
a **grid_limit** of `1024` and a **block_size** of `256` could be a decent
choice. Interestingly, the *sequential sum* suffers from **precision issue**
when using the **float** datatype, while the *CUDA based sum* does not.

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
# # Elements 1e+07
# [00014.215 ms] [15.403683] sumSeq
# [00000.087 ms] [16.695320] sumCuda<<<1024, 32>>>
# [00000.079 ms] [16.695307] sumCuda<<<1024, 64>>>
# [00000.079 ms] [16.695312] sumCuda<<<1024, 128>>>
# [00000.079 ms] [16.695326] sumCuda<<<1024, 256>>>
# [00000.080 ms] [16.695335] sumCuda<<<1024, 512>>>
# [00000.081 ms] [16.695333] sumCuda<<<1024, 1024>>>
# [00000.095 ms] [16.695314] sumCuda<<<2048, 32>>>
# [00000.091 ms] [16.695311] sumCuda<<<2048, 64>>>
# [00000.083 ms] [16.695307] sumCuda<<<2048, 128>>>
# [00000.084 ms] [16.695290] sumCuda<<<2048, 256>>>
# [00000.082 ms] [16.695299] sumCuda<<<2048, 512>>>
# [00000.094 ms] [16.695299] sumCuda<<<2048, 1024>>>
# [00000.097 ms] [16.695293] sumCuda<<<4096, 32>>>
# [00000.088 ms] [16.695303] sumCuda<<<4096, 64>>>
# [00000.089 ms] [16.695318] sumCuda<<<4096, 128>>>
# [00000.088 ms] [16.695309] sumCuda<<<4096, 256>>>
# [00000.089 ms] [16.695292] sumCuda<<<4096, 512>>>
# [00000.143 ms] [16.695297] sumCuda<<<4096, 1024>>>
# [00000.099 ms] [16.695312] sumCuda<<<8192, 32>>>
# [00000.098 ms] [16.695303] sumCuda<<<8192, 64>>>
# [00000.095 ms] [16.695307] sumCuda<<<8192, 128>>>
# [00000.098 ms] [16.695282] sumCuda<<<8192, 256>>>
# [00000.129 ms] [16.695316] sumCuda<<<8192, 512>>>
# [00000.220 ms] [16.695295] sumCuda<<<8192, 1024>>>
# [00000.117 ms] [16.695354] sumCuda<<<16384, 32>>>
# [00000.114 ms] [16.695337] sumCuda<<<16384, 64>>>
# [00000.113 ms] [16.695377] sumCuda<<<16384, 128>>>
# [00000.140 ms] [16.695253] sumCuda<<<16384, 256>>>
# [00000.201 ms] [16.695305] sumCuda<<<16384, 512>>>
# [00000.246 ms] [16.695335] sumCuda<<<16384, 1024>>>
# [00000.165 ms] [16.695353] sumCuda<<<32768, 32>>>
# [00000.158 ms] [16.695341] sumCuda<<<32768, 64>>>
# [00000.171 ms] [16.695211] sumCuda<<<32768, 128>>>
# [00000.226 ms] [16.695206] sumCuda<<<32768, 256>>>
# [00000.226 ms] [16.695301] sumCuda<<<32768, 512>>>
# [00000.245 ms] [16.695335] sumCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/EseR7Oa.gif)][sheetp]
[![](https://i.imgur.com/8TCXC0q.gif)][sheetp]

[![](https://i.imgur.com/AmQEavQ.gif)][sheetp]
[![](https://i.imgur.com/U9YlVC4.gif)][sheetp]

[![](https://i.imgur.com/ZLqcgXy.gif)][sheetp]
[![](https://i.imgur.com/HOx7ppf.gif)][sheetp]

[![](https://i.imgur.com/OU4iUc5.gif)][sheetp]
[![](https://i.imgur.com/yDIzLIW.gif)][sheetp]

[![](https://i.imgur.com/riXcPkR.png)][sheetp]
[![](https://i.imgur.com/5FGuvPS.png)][sheetp]
[![](https://i.imgur.com/mk6BUxi.png)][sheetp]
[![](https://i.imgur.com/3hX2a05.png)][sheetp]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)

<br>
<br>

[![](https://i.imgur.com/ulq4FzL.jpg)](https://www.youtube.com/watch?v=vTdodyhhjww)
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/375230047.svg)](https://zenodo.org/badge/latestdoi/375230047)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[Nsight Compute]: https://developer.nvidia.com/nsight-compute
[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
[charts]: https://photos.app.goo.gl/Jytw1qgSFPoTrL1FA
[sheets]: https://docs.google.com/spreadsheets/d/1jNNg43h19DUNwdwQVkOsLe5Dq-f_MfIbzaGG1pai4SU/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vQ4uC8kGvwbHyfRM-YpIHNEdWHwb7ufNGN2vxjEj1qFmjCmtrGigpTWs8SukrEh9iv7t20ZaQqXGa-0/pubhtml
