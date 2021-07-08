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
**128** could be a decent choice.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. This
experiment was done with guidance from [Prof. Dip Sankar Banerjee] and
[Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+09
# [01384.584 ms] [15.403683] sumSeq
# [00005.794 ms] [21.299603] sumCuda<<<1024, 32>>>
# [00004.818 ms] [21.300037] sumCuda<<<1024, 64>>>
# [00004.508 ms] [21.300262] sumCuda<<<1024, 128>>>
# [00004.528 ms] [21.300373] sumCuda<<<1024, 256>>>
# [00004.623 ms] [21.300426] sumCuda<<<1024, 512>>>
# [00004.525 ms] [21.300455] sumCuda<<<1024, 1024>>>
# [00006.373 ms] [21.300037] sumCuda<<<2048, 32>>>
# [00005.414 ms] [21.300262] sumCuda<<<2048, 64>>>
# [00004.532 ms] [21.300373] sumCuda<<<2048, 128>>>
# [00004.620 ms] [21.300426] sumCuda<<<2048, 256>>>
# [00004.504 ms] [21.300457] sumCuda<<<2048, 512>>>
# [00004.496 ms] [21.300468] sumCuda<<<2048, 1024>>>
# [00005.374 ms] [21.300262] sumCuda<<<4096, 32>>>
# [00004.875 ms] [21.300371] sumCuda<<<4096, 64>>>
# [00004.619 ms] [21.300428] sumCuda<<<4096, 128>>>
# [00004.524 ms] [21.300453] sumCuda<<<4096, 256>>>
# [00004.494 ms] [21.300470] sumCuda<<<4096, 512>>>
# [00004.494 ms] [21.300476] sumCuda<<<4096, 1024>>>
# [00004.812 ms] [21.300371] sumCuda<<<8192, 32>>>
# [00004.591 ms] [21.300426] sumCuda<<<8192, 64>>>
# [00004.507 ms] [21.300455] sumCuda<<<8192, 128>>>
# [00004.493 ms] [21.300468] sumCuda<<<8192, 256>>>
# [00004.491 ms] [21.300476] sumCuda<<<8192, 512>>>
# [00004.489 ms] [21.300478] sumCuda<<<8192, 1024>>>
# [00004.678 ms] [21.300430] sumCuda<<<16384, 32>>>
# [00004.504 ms] [21.300455] sumCuda<<<16384, 64>>>
# [00004.495 ms] [21.300468] sumCuda<<<16384, 128>>>
# [00004.495 ms] [21.300474] sumCuda<<<16384, 256>>>
# [00004.488 ms] [21.300478] sumCuda<<<16384, 512>>>
# [00004.489 ms] [21.300478] sumCuda<<<16384, 1024>>>
# [00004.646 ms] [21.300453] sumCuda<<<32768, 32>>>
# [00004.520 ms] [21.300468] sumCuda<<<32768, 64>>>
# [00004.494 ms] [21.300476] sumCuda<<<32768, 128>>>
# [00004.488 ms] [21.300480] sumCuda<<<32768, 256>>>
# [00004.488 ms] [21.300478] sumCuda<<<32768, 512>>>
# [00004.490 ms] [21.300482] sumCuda<<<32768, 1024>>>
```

[![](https://i.imgur.com/CWySswQ.gif)][sheets]
[![](https://i.imgur.com/o3mYdbR.gif)][sheets]
[![](https://i.imgur.com/jGqYBwP.gif)][sheets]
[![](https://i.imgur.com/ktH8eSd.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)

<br>
<br>

[![](https://i.imgur.com/SrEEKD5.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[charts]: https://photos.app.goo.gl/795Rcbqa14srjoZBA
[sheets]: https://docs.google.com/spreadsheets/d/1pgIn6dcrKtVv0SoaJeQwTe1CzHRKuoUOXjn5_KJqrA8/edit?usp=sharing
