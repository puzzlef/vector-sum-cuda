Performance of *sequential* vs *CUDA-based* **vector element sum**.

This experiment was for comparing the performance between:
1. Find `sum(x)` using a single thread (**sequential**).
2. Find `sum(x)` accelerated using **CUDA** (*not power-of-2* reduce).
3. Find `sum(x)` accelerated using **CUDA** (*power-of-2* reduce).

Here `x` is a 32-bit integer vector. Both approaches were attempted on a number
of vector sizes, running each approach 5 times per size to get a good time
measure. Note that time taken to copy data back and forth from the GPU is not
measured, and the sequential approach does not make use of *SIMD instructions*.
While it might seem that **CUDA** approach would be a clear winner, the results
indicate it is dependent upon the workload. Results indicate that **from 10^5**
**elements, CUDA approach performs better** than sequential. Both CUDA approaches
(*not power-of-2*/*power-of-2* reduce) seem to have similar performance.

All outputs are saved in a [gist] and a small part of the output is listed here.
Some [charts] are also included below, generated from [sheets]. This experiment
was done with guidance from [Prof. Kishore Kothapalli] and
[Prof. Dip Sankar Banerjee].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# [00000.002 ms; 1e+03 elems.] [502942114] sumSeq
# [00001.128 ms; 1e+03 elems.] [502942114] sumCuda
# [00000.018 ms; 1e+03 elems.] [502942114] sumCudaPow2
# ...
```

[![](https://i.imgur.com/WAY6rGl.png)][sheetp]

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

<br>
<br>

[![](https://i.imgur.com/MOJPoM0.jpg)](https://www.youtube.com/watch?v=E0_Ic1P-Hzg)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[gist]: https://gist.github.com/wolfram77/44465db42bf17b0464159331388da526
[charts]: https://imgur.com/a/bnRHipj
[sheets]: https://docs.google.com/spreadsheets/d/19hBlJQv7JwEuoA2X0aw5IS0_MghifQfr3TG_WgmoSww/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTwzwsCzU25d7YEo6kVST5tRVSWKESczT7Wo51ML_tghIrBlOa4e9IrCgeG5c5_lOM5Ojzu8Txq8xjQ/pubhtml
