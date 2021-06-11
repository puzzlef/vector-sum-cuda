Comparing various launch configs for CUDA based vector element sum (in-place).

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. This is an in-place sum, meaning the single sum values is
calculated entirely by the GPU. This is done using either 2 (if `grid_limit`
is `1024`) or 3 kernel calls (otherwise). A `block_size` of `128`
(decent choice for sum) is used for *2nd* kernel, if there are 3 kernels.
Sum here represents any reduction operation that processes several values
to a single value. Using a `grid_limit` of **1024** and a `block_size` of
**256** could be a decent choice.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+09
# [01423.561 ms] [15.403683] sumSeq
# [00000.012 ms] [21.299603] sumCuda<<<1024, 32>>>
# [00000.011 ms] [21.300037] sumCuda<<<1024, 64>>>
# [00000.014 ms] [21.300262] sumCuda<<<1024, 128>>>
# [00000.011 ms] [21.300373] sumCuda<<<1024, 256>>>
# [00000.012 ms] [21.300426] sumCuda<<<1024, 512>>>
# [00000.017 ms] [21.300455] sumCuda<<<1024, 1024>>>
# [00000.012 ms] [21.300037] sumCuda<<<2048, 32>>>
# [00000.012 ms] [21.300262] sumCuda<<<2048, 64>>>
# [00000.021 ms] [21.300373] sumCuda<<<2048, 128>>>
# [00000.014 ms] [21.300426] sumCuda<<<2048, 256>>>
# [00000.015 ms] [21.300457] sumCuda<<<2048, 512>>>
# [00000.013 ms] [21.300468] sumCuda<<<2048, 1024>>>
# [00000.012 ms] [21.300262] sumCuda<<<4096, 32>>>
# [00000.020 ms] [21.300371] sumCuda<<<4096, 64>>>
# [00000.014 ms] [21.300428] sumCuda<<<4096, 128>>>
# [00000.014 ms] [21.300453] sumCuda<<<4096, 256>>>
# [00000.013 ms] [21.300470] sumCuda<<<4096, 512>>>
# [00000.011 ms] [21.300476] sumCuda<<<4096, 1024>>>
# [00000.020 ms] [21.300371] sumCuda<<<8192, 32>>>
# [00000.014 ms] [21.300426] sumCuda<<<8192, 64>>>
# [00000.015 ms] [21.300455] sumCuda<<<8192, 128>>>
# [00000.013 ms] [21.300468] sumCuda<<<8192, 256>>>
# [00000.016 ms] [21.300476] sumCuda<<<8192, 512>>>
# [00000.023 ms] [21.300478] sumCuda<<<8192, 1024>>>
# [00000.014 ms] [21.300430] sumCuda<<<16384, 32>>>
# [00000.014 ms] [21.300455] sumCuda<<<16384, 64>>>
# [00000.013 ms] [21.300468] sumCuda<<<16384, 128>>>
# [00000.015 ms] [21.300474] sumCuda<<<16384, 256>>>
# [00000.014 ms] [21.300478] sumCuda<<<16384, 512>>>
# [00000.023 ms] [21.300478] sumCuda<<<16384, 1024>>>
# [00000.016 ms] [21.300453] sumCuda<<<32768, 32>>>
# [00000.013 ms] [21.300468] sumCuda<<<32768, 64>>>
# [00000.015 ms] [21.300476] sumCuda<<<32768, 128>>>
# [00000.014 ms] [21.300480] sumCuda<<<32768, 256>>>
# [00000.015 ms] [21.300478] sumCuda<<<32768, 512>>>
# [00000.014 ms] [21.300482] sumCuda<<<32768, 1024>>>
```

[![](https://i.imgur.com/M3FPYUc.gif)][sheets]
[![](https://i.imgur.com/zT3uKyH.gif)][sheets]
[![](https://i.imgur.com/B74KDHR.gif)][sheets]
[![](https://i.imgur.com/d7HKk6K.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)

<br>
<br>

[![](https://i.imgur.com/cbgBvPJ.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[charts]: https://photos.app.goo.gl/795Rcbqa14srjoZBA
[sheets]: https://docs.google.com/spreadsheets/d/1pgIn6dcrKtVv0SoaJeQwTe1CzHRKuoUOXjn5_KJqrA8/edit?usp=sharing
