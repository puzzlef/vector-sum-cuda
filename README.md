Comparing performance of *sequential* vs *CUDA-based* **vector element sum**.

We take a floating-point vector `x`, with number of *elements* ranging from
`1E+6` to `1E+9`, and sum them up using CUDA (`Σx`). We attempt each element
count with various approaches, running each approach 5 times to get a good time
measure. Sum here represents any `reduce()` operation that processes several
values to a single value. I thank the guidance from [Prof. Kishore Kothapalli] and
[Prof. Dip Sankar Banerjee].

<br>


### Adjusting launch config for Memcpy approach

In this experiment ([memcpy-adust-launch]), we compare various *launch configs*
for *CUDA-based* vector element sum, using the **memcpy** approach. We attempt
different element counts with various *CUDA launch configs*.

This sum uses *memcpy* to transfer partial results to CPU, where the final sum
is calculated. If the result can be used within GPU itself, it *might* be faster
to calculate complete sum [in-place] instead of transferring to CPU. Results
indicate that a **grid_limit** of `1024` and a **block_size** of `128/256` is
suitable for **float** datatype, and a **grid_limit** of `1024` and a
**block_size** of `256` is suitable for **double** datatype. Thus, using a
**grid_limit** of `1024` and a **block_size** of `256` could be a decent choice.
Interestingly, the *sequential sum* suffers from **precision issue** when using
the **float** datatype, while the *CUDA based sum* does not.

[memcpy-adust-launch]: https://github.com/puzzlef/vector-sum-cuda/tree/memcpy-adjust-launch

<br>


### Adjust per-thread duty for Memcpy approach

In this experiment ([memcpy-adjust-duty]), we compare various *per-thread duty*
*numbers* for **CUDA based vector element sum (memcpy)**. Here, we attempt each
element count with various **CUDA launch configs**, and **per-thread-duties**.
Rest of the experimental setup is similar to the [memcpy-adjust-launch]
experiment. Results indicate no significant difference between
[memcpy launch][memcpy-adust-launch] approach, and this one.

[memcpy-adjust-duty]: https://github.com/puzzlef/vector-sum-cuda/tree/memcpy-adjust-duty

<br>


### Adjusting launch config for Inplace approach

In this experiment ([inplace-adjust-launch]), we compare various *launch*
*configs* for CUDA based **vector element sum** (**in-place**) (in-place). We
attempt different element counts with various **CUDA** **launch configs**. This
is an in-place sum, meaning the single sum values is calculated entirely by the
GPU. This is done using 2 kernel calls.

A number of possible optimizations including *multiple reads per loop*
*iteration*, *loop unrolled reduce*, *atomic adds*, and *multiple kernels*
provided no benefit (see [branches]). A simple **one read per loop iteration**
and **standard reduce loop** (minimizing warp divergence) is both **shorter**
and **works best**. For **float**, a **grid_limit** of `1024` and a
**block_size** of `128` is a decent choice. For **double**, a **grid_limit** of
`1024` and a **block_size** of `256` is a decent choice. Interestingly, the
*sequential sum* suffers from **precision issue** when using the **float**
datatype, while the *CUDA based sum* does not (just like with
[memcpy sum][memcpy-adust-launch]).

[inplace-adjust-launch]: https://github.com/puzzlef/vector-sum-cuda/tree/inplace-adjust-launch

<br>


### Comparison of Memcpy and Inplace approach

In this experiment ([memcpy-vs-inplace]), we compare the performance of
[memcpy][memcpy-adust-launch] vs [in-place][inplace-adjust-launch] based CUDA
based **vector element sum**. It appears **both** **approaches** have
**similar** performance.

[memcpy-vs-inplace]: https://github.com/puzzlef/vector-sum-cuda/tree/memcpy-vs-inplace

<br>


### Comparison with Sequential approach

In this experiment ([compare-sequential], [main]), we compare the performance
between finding `sum(x)` using a single thread (**sequential**) and **CUDA**
(*not power-of-2* and *power-of-2* reduce). Here `x` is a 32-bit integer vector.
We attempt the approaches on a number of vector sizes. Note that time taken to
copy data back and forth from the GPU is not measured, and the sequential
approach does not make use of *SIMD instructions*.

While it might seem that **CUDA** approach would be a clear winner, the results
indicate it is dependent upon the workload. Results indicate that **from 10^5**
**elements, CUDA approach performs better** than sequential. Both CUDA
approaches (*not power-of-2*/*power-of-2* reduce) seem to have similar
performance. All outputs are saved in a [gist]. Some [charts] are also included
below, generated from [sheets].

[![](https://i.imgur.com/WAY6rGl.png)][sheetp]

[compare-sequential]: https://github.com/puzzlef/vector-sum-cuda/tree/compare-sequential
[main]: https://github.com/puzzlef/vector-sum-cuda

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://gist.github.com/wolfram77/72c51e494eaaea1c21a9c4021ad0f320)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [How to enable C++17 code generation in VS2019 CUDA project](https://stackoverflow.com/a/63057409/1413259)
- ["More than one operator + matches these operands" error](https://stackoverflow.com/a/10343618/1413259)
- [How to import VSCode keybindings into Visual Studio?](https://stackoverflow.com/a/62417446/1413259)
- [Explicit conversion constructors (C++ only)](https://www.ibm.com/docs/en/i/7.3?topic=only-explicit-conversion-constructors-c)
- [Configure X11 Forwarding with PuTTY and Xming](https://www.centlinux.com/2019/01/configure-x11-forwarding-putty-xming-windows.html)
- [code-server setup and configuration](https://coder.com/docs/code-server/latest/guide)
- [Installing snap on CentOS](https://snapcraft.io/docs/installing-snap-on-centos)
- [Git pulling a branch from another repository?](https://stackoverflow.com/a/46289324/1413259)

<br>
<br>


[![](https://i.imgur.com/MOJPoM0.jpg)](https://www.youtube.com/watch?v=E0_Ic1P-Hzg)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/558223916.svg)](https://zenodo.org/badge/latestdoi/558223916)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[branches]: https://github.com/puzzlef/vector-sum-cuda/branches
[gist]: https://gist.github.com/wolfram77/44465db42bf17b0464159331388da526
[charts]: https://imgur.com/a/bnRHipj
[sheets]: https://docs.google.com/spreadsheets/d/19hBlJQv7JwEuoA2X0aw5IS0_MghifQfr3TG_WgmoSww/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTwzwsCzU25d7YEo6kVST5tRVSWKESczT7Wo51ML_tghIrBlOa4e9IrCgeG5c5_lOM5Ojzu8Txq8xjQ/pubhtml
