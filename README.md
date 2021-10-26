Comparing various *per-thread duty numbers* for **CUDA based vector element sum (memcpy)**.

A floating-point vector `x`, with number of **elements** from `1E+6` to
`1E+9` was summed up using CUDA (`Σx`). Each element count was attempted with
various **CUDA launch configs** and **per-thread-duties**, running each config
5 times to get a good time measure. Sum here represents any `reduce()`
operation that processes several values to a single value.

This sum uses *memcpy* to transfer partial results to CPU, where the final sum
is calculated. If the result can be used within GPU itself, it *might* be
faster to calculate complete sum [in-place] instead of transferring to CPU.
Results indicate no significant difference between [launch adjust] approach,
and this one.

All outputs are saved in [out](out/) and a small part of the output is listed
here. This experiment was done with guidance from [Prof. Dip Sankar Banerjee]
and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# ...
#
# # Elements 1e+07
# [00014.537 ms] [15.403683] sumSeq
# [00000.317 ms] [16.696144] sumCuda<<<auto, 32>>> [thread-duty=1]
# [00000.295 ms] [16.696144] sumCuda<<<auto, 32>>> [thread-duty=2]
# [00000.295 ms] [16.696144] sumCuda<<<auto, 32>>> [thread-duty=3]
# [00000.294 ms] [16.696144] sumCuda<<<auto, 32>>> [thread-duty=4]
# [00000.238 ms] [16.695305] sumCuda<<<auto, 32>>> [thread-duty=6]
# [00000.185 ms] [16.695381] sumCuda<<<auto, 32>>> [thread-duty=8]
# [00000.138 ms] [16.695335] sumCuda<<<auto, 32>>> [thread-duty=12]
# [00000.127 ms] [16.695259] sumCuda<<<auto, 32>>> [thread-duty=16]
# [00000.109 ms] [16.695305] sumCuda<<<auto, 32>>> [thread-duty=24]
# [00000.105 ms] [16.695305] sumCuda<<<auto, 32>>> [thread-duty=32]
# [00000.094 ms] [16.695299] sumCuda<<<auto, 32>>> [thread-duty=48]
# [00000.089 ms] [16.695320] sumCuda<<<auto, 32>>> [thread-duty=64]
# [00000.297 ms] [16.694933] sumCuda<<<auto, 64>>> [thread-duty=1]
# [00000.294 ms] [16.694933] sumCuda<<<auto, 64>>> [thread-duty=2]
# [00000.240 ms] [16.695873] sumCuda<<<auto, 64>>> [thread-duty=3]
# [00000.185 ms] [16.695036] sumCuda<<<auto, 64>>> [thread-duty=4]
# [00000.139 ms] [16.695379] sumCuda<<<auto, 64>>> [thread-duty=6]
# [00000.123 ms] [16.695301] sumCuda<<<auto, 64>>> [thread-duty=8]
# [00000.107 ms] [16.695312] sumCuda<<<auto, 64>>> [thread-duty=12]
# [00000.101 ms] [16.695330] sumCuda<<<auto, 64>>> [thread-duty=16]
# [00000.093 ms] [16.695316] sumCuda<<<auto, 64>>> [thread-duty=24]
# [00000.092 ms] [16.695295] sumCuda<<<auto, 64>>> [thread-duty=32]
# [00000.085 ms] [16.695314] sumCuda<<<auto, 64>>> [thread-duty=48]
# [00000.084 ms] [16.695316] sumCuda<<<auto, 64>>> [thread-duty=64]
# [00000.303 ms] [16.694792] sumCuda<<<auto, 128>>> [thread-duty=1]
# [00000.198 ms] [16.695229] sumCuda<<<auto, 128>>> [thread-duty=2]
# [00000.147 ms] [16.695190] sumCuda<<<auto, 128>>> [thread-duty=3]
# [00000.124 ms] [16.695284] sumCuda<<<auto, 128>>> [thread-duty=4]
# [00000.108 ms] [16.695293] sumCuda<<<auto, 128>>> [thread-duty=6]
# [00000.101 ms] [16.695303] sumCuda<<<auto, 128>>> [thread-duty=8]
# [00000.094 ms] [16.695312] sumCuda<<<auto, 128>>> [thread-duty=12]
# [00000.091 ms] [16.695297] sumCuda<<<auto, 128>>> [thread-duty=16]
# [00000.086 ms] [16.695311] sumCuda<<<auto, 128>>> [thread-duty=24]
# [00000.087 ms] [16.695332] sumCuda<<<auto, 128>>> [thread-duty=32]
# [00000.081 ms] [16.695333] sumCuda<<<auto, 128>>> [thread-duty=48]
# [00000.080 ms] [16.695314] sumCuda<<<auto, 128>>> [thread-duty=64]
# [00000.259 ms] [16.695635] sumCuda<<<auto, 256>>> [thread-duty=1]
# [00000.152 ms] [16.695305] sumCuda<<<auto, 256>>> [thread-duty=2]
# [00000.126 ms] [16.695290] sumCuda<<<auto, 256>>> [thread-duty=3]
# [00000.101 ms] [16.695293] sumCuda<<<auto, 256>>> [thread-duty=4]
# [00000.093 ms] [16.695322] sumCuda<<<auto, 256>>> [thread-duty=6]
# [00000.090 ms] [16.695301] sumCuda<<<auto, 256>>> [thread-duty=8]
# [00000.086 ms] [16.695305] sumCuda<<<auto, 256>>> [thread-duty=12]
# [00000.084 ms] [16.695326] sumCuda<<<auto, 256>>> [thread-duty=16]
# [00000.082 ms] [16.695318] sumCuda<<<auto, 256>>> [thread-duty=24]
# [00000.082 ms] [16.695305] sumCuda<<<auto, 256>>> [thread-duty=32]
# [00000.082 ms] [16.695309] sumCuda<<<auto, 256>>> [thread-duty=48]
# [00000.080 ms] [16.695320] sumCuda<<<auto, 256>>> [thread-duty=64]
# [00000.231 ms] [16.695301] sumCuda<<<auto, 512>>> [thread-duty=1]
# [00000.139 ms] [16.695326] sumCuda<<<auto, 512>>> [thread-duty=2]
# [00000.117 ms] [16.695290] sumCuda<<<auto, 512>>> [thread-duty=3]
# [00000.089 ms] [16.695305] sumCuda<<<auto, 512>>> [thread-duty=4]
# [00000.086 ms] [16.695314] sumCuda<<<auto, 512>>> [thread-duty=6]
# [00000.084 ms] [16.695305] sumCuda<<<auto, 512>>> [thread-duty=8]
# [00000.081 ms] [16.695337] sumCuda<<<auto, 512>>> [thread-duty=12]
# [00000.081 ms] [16.695314] sumCuda<<<auto, 512>>> [thread-duty=16]
# [00000.081 ms] [16.695328] sumCuda<<<auto, 512>>> [thread-duty=24]
# [00000.080 ms] [16.695307] sumCuda<<<auto, 512>>> [thread-duty=32]
# [00000.081 ms] [16.695309] sumCuda<<<auto, 512>>> [thread-duty=48]
# [00000.078 ms] [16.695320] sumCuda<<<auto, 512>>> [thread-duty=64]
# [00000.247 ms] [16.695335] sumCuda<<<auto, 1024>>> [thread-duty=1]
# [00000.153 ms] [16.695295] sumCuda<<<auto, 1024>>> [thread-duty=2]
# [00000.121 ms] [16.695328] sumCuda<<<auto, 1024>>> [thread-duty=3]
# [00000.096 ms] [16.695316] sumCuda<<<auto, 1024>>> [thread-duty=4]
# [00000.087 ms] [16.695326] sumCuda<<<auto, 1024>>> [thread-duty=6]
# [00000.082 ms] [16.695328] sumCuda<<<auto, 1024>>> [thread-duty=8]
# [00000.079 ms] [16.695309] sumCuda<<<auto, 1024>>> [thread-duty=12]
# [00000.079 ms] [16.695295] sumCuda<<<auto, 1024>>> [thread-duty=16]
# [00000.081 ms] [16.695311] sumCuda<<<auto, 1024>>> [thread-duty=24]
# [00000.080 ms] [16.695311] sumCuda<<<auto, 1024>>> [thread-duty=32]
# [00000.083 ms] [16.695311] sumCuda<<<auto, 1024>>> [thread-duty=48]
# [00000.079 ms] [16.695318] sumCuda<<<auto, 1024>>> [thread-duty=64]
#
# ...
```

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)

<br>
<br>

[![](https://i.imgur.com/KExwVG1.jpg)](https://www.youtube.com/watch?v=A7TKQKAFIi4)
[![DOI](https://zenodo.org/badge/413857764.svg)](https://zenodo.org/badge/latestdoi/413857764)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[launch adjust]: https://github.com/puzzlef/sum-cuda-memcpy-adjust-launch
[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
