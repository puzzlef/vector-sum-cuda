Performance of [memcpy] vs [in-place] based CUDA based vector element sum.

A floating-point vector `x`, with no. of **elements** `1E+6` to `1E+9` was
summed up using CUDA (`Σx`). Each no. of elements was attempted with both
approaches, running each approach 5 times to get a good time measure. Sum
here represents any reduction operation that processes several values to a
single value. If the sum result needs to be used by the GPU in a further
step, **in-place** sum is clearly **much faster** than *memcpy* approach.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# # Elements 1e+06
# [00001.816 ms] [14.392727] sumSeq
# [00000.040 ms] [14.392727] sumMemcpyCuda
# [00000.009 ms] [14.392727] sumInplaceCuda
#
# # Elements 1e+07
# [00015.018 ms] [16.695311] sumSeq
# [00000.122 ms] [16.695311] sumMemcpyCuda
# [00000.011 ms] [16.695311] sumInplaceCuda
#
# # Elements 1e+08
# [00159.594 ms] [18.997896] sumSeq
# [00001.002 ms] [18.997896] sumMemcpyCuda
# [00000.011 ms] [18.997896] sumInplaceCuda
#
# # Elements 1e+09
# [01611.637 ms] [21.300482] sumSeq
# [00009.978 ms] [21.300482] sumMemcpyCuda
# [00000.035 ms] [21.300482] sumInplaceCuda
```

[![](https://i.imgur.com/LzKB9Gh.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)

<br>
<br>

[![](https://i.imgur.com/ZCtUOgH.jpg)](https://www.youtube.com/watch?v=-DynbmAehL8)

[memcpy]: https://github.com/puzzlef/sum-cuda-memcpy-adjust-launch
[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
[charts]: https://photos.app.goo.gl/a8PM8K1FXPm1LQed8
[sheets]: https://docs.google.com/spreadsheets/d/1CpZRcOcQ1FKTX0nLWb6R7znPtgeQXhlA7HNEIm2_ZRc/edit?usp=sharing
