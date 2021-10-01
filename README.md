Performance of [memcpy] vs [in-place] based CUDA based **vector element sum**.

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with both
approaches, running each approach 5 times to get a good time measure. Sum
here represents any reduction operation that processes several values to a
single value. It appears **both approaches** have **similar** performance.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. This
experiment was done with guidance from [Prof. Dip Sankar Banerjee] and
[Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# (DOUBLE)
# # Elements 1e+06
# [00002.003 ms] [14.392727] sumSeq
# [00000.039 ms] [14.392727] sumMemcpyCuda
# [00000.034 ms] [14.392727] sumInplaceCuda
#
# # Elements 1e+07
# [00014.767 ms] [16.695311] sumSeq
# [00000.123 ms] [16.695311] sumMemcpyCuda
# [00000.116 ms] [16.695311] sumInplaceCuda
#
# # Elements 1e+08
# [00151.176 ms] [18.997896] sumSeq
# [00000.940 ms] [18.997896] sumMemcpyCuda
# [00000.935 ms] [18.997896] sumInplaceCuda
#
# # Elements 1e+09
# [01444.112 ms] [21.300482] sumSeq
# [00008.990 ms] [21.300482] sumMemcpyCuda
# [00008.989 ms] [21.300482] sumInplaceCuda
```

[![](https://i.imgur.com/rJNTBF3.gif)][sheetp]

[![](https://i.imgur.com/YvBEioN.png)][sheetp]
[![](https://i.imgur.com/m2G6WgE.png)][sheetp]
[![](https://i.imgur.com/04VUuVM.png)][sheetp]
[![](https://i.imgur.com/gDrnsNp.png)][sheetp]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)
- [Unspecified launch failure on Memcpy](https://stackoverflow.com/a/27278218/1413259)

<br>
<br>

[![](https://i.imgur.com/FIv7piL.jpg)](https://www.youtube.com/watch?v=Zf8xRNO1xIU)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[memcpy]: https://github.com/puzzlef/sum-cuda-memcpy-adjust-launch
[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
[charts]: https://photos.app.goo.gl/a8PM8K1FXPm1LQed8
[sheets]: https://docs.google.com/spreadsheets/d/1CpZRcOcQ1FKTX0nLWb6R7znPtgeQXhlA7HNEIm2_ZRc/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vRpLHagCPAIPGmT43G6n8-U3S3u3vp04_M4ky8ekfG9U_MOxAtSSUZVFIM6msmYgYies4Ftsmm-DAeh/pubhtml
