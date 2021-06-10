Comparing various launch configs for CUDA based vector element sum.

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. Sum here represents any reduction operation that processes
several values to a single value. Using a `grid_limit` of **1024** and a
`block_size` of **128** could be a decent choice. This sum uses *memcpy*
to transfer partial results to CPU, where the final sum is calculated. If
the result can be used within GPU itself, it might be faster to calculate
complete sum [in-place] instead of transferring to CPU.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+09
# [01418.760 ms] [15.403683] sumSeq
# [00005.728 ms] [21.299604] sumCuda<<<1024, 32>>>
# [00004.832 ms] [21.300009] sumCuda<<<1024, 64>>>
# [00004.507 ms] [21.300274] sumCuda<<<1024, 128>>>
# [00004.550 ms] [21.300369] sumCuda<<<1024, 256>>>
# [00004.653 ms] [21.300442] sumCuda<<<1024, 512>>>
# [00004.509 ms] [21.300453] sumCuda<<<1024, 1024>>>
# [00006.389 ms] [21.300039] sumCuda<<<2048, 32>>>
# [00005.458 ms] [21.300262] sumCuda<<<2048, 64>>>
# [00004.528 ms] [21.300379] sumCuda<<<2048, 128>>>
# [00004.628 ms] [21.300423] sumCuda<<<2048, 256>>>
# [00004.514 ms] [21.300453] sumCuda<<<2048, 512>>>
# [00004.501 ms] [21.300432] sumCuda<<<2048, 1024>>>
# [00005.386 ms] [21.300251] sumCuda<<<4096, 32>>>
# [00004.886 ms] [21.300367] sumCuda<<<4096, 64>>>
# [00004.672 ms] [21.300426] sumCuda<<<4096, 128>>>
# [00004.521 ms] [21.300470] sumCuda<<<4096, 256>>>
# [00004.506 ms] [21.300474] sumCuda<<<4096, 512>>>
# [00004.506 ms] [21.300486] sumCuda<<<4096, 1024>>>
# [00004.823 ms] [21.300383] sumCuda<<<8192, 32>>>
# [00004.609 ms] [21.300419] sumCuda<<<8192, 64>>>
# [00004.525 ms] [21.300449] sumCuda<<<8192, 128>>>
# [00004.516 ms] [21.300476] sumCuda<<<8192, 256>>>
# [00004.524 ms] [21.300457] sumCuda<<<8192, 512>>>
# [00004.518 ms] [21.300510] sumCuda<<<8192, 1024>>>
# [00004.743 ms] [21.300440] sumCuda<<<16384, 32>>>
# [00004.547 ms] [21.300442] sumCuda<<<16384, 64>>>
# [00004.537 ms] [21.300453] sumCuda<<<16384, 128>>>
# [00004.538 ms] [21.300501] sumCuda<<<16384, 256>>>
# [00004.547 ms] [21.300495] sumCuda<<<16384, 512>>>
# [00004.533 ms] [21.300480] sumCuda<<<16384, 1024>>>
# [00004.735 ms] [21.300377] sumCuda<<<32768, 32>>>
# [00004.598 ms] [21.300343] sumCuda<<<32768, 64>>>
# [00004.569 ms] [21.300446] sumCuda<<<32768, 128>>>
# [00004.565 ms] [21.300461] sumCuda<<<32768, 256>>>
# [00004.565 ms] [21.300476] sumCuda<<<32768, 512>>>
# [00004.579 ms] [21.300436] sumCuda<<<32768, 1024>>>
```

[![](https://i.imgur.com/U9U5Vbm.gif)][sheets]
[![](https://i.imgur.com/lDw866b.gif)][sheets]
[![](https://i.imgur.com/5JsrM6E.gif)][sheets]
[![](https://i.imgur.com/CAJ8wwT.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)

<br>
<br>

[![](https://i.imgur.com/SrEEKD5.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
[charts]: https://photos.app.goo.gl/Jytw1qgSFPoTrL1FA
[sheets]: https://docs.google.com/spreadsheets/d/1F2L2ro1rOO-ZSNcXnhVDJqltwNfg2lSoh8WlqQ-YMw8/edit?usp=sharing
