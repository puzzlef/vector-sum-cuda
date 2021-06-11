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
```

[![](https://i.imgur.com/U9U5Vbm.gif)][sheets]

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
