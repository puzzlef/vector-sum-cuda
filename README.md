Comparing various launch configs for CUDA based vector multiply.

Two floating-point vector `x` and `y`, with no. of **elements** `1E+6` to
`1E+9` were multiplied using CUDA. Each no. of elements was attempted with
various **CUDA launch configs**, running each config 5 times to get a good
time measure. Multiplication here represents any memory-aligned independent
operation. Using a **large** `grid_limit` and a `block_size` of **256** could
be a decent choice.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
```

[![](https://i.imgur.com/bGUUPot.gif)][sheets]
[![](https://i.imgur.com/IagoPuk.gif)][sheets]
[![](https://i.imgur.com/tCUuW0a.gif)][sheets]
[![](https://i.imgur.com/U6jbPeH.gif)][sheets]

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)

<br>
<br>

[![](https://i.imgur.com/lRwvZLe.png)](https://www.youtube.com/watch?v=vTdodyhhjww)

[charts]: https://photos.app.goo.gl/xorYb1MZSNqxUgNy7
[sheets]: https://docs.google.com/spreadsheets/d/1fWcVNQbANgiNepryktAsIWUHCNiAi-Yf1qQyiLsTJio/edit?usp=sharing
