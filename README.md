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
# [00014.521 ms] [15.403683] sumSeq
# [00000.315 ms] [16.695896] sumCuda<<<auto, 32>>> [thread-duty=1]
# [00000.295 ms] [16.695896] sumCuda<<<auto, 32>>> [thread-duty=2]
# [00000.294 ms] [16.695896] sumCuda<<<auto, 32>>> [thread-duty=3]
# [00000.292 ms] [16.695896] sumCuda<<<auto, 32>>> [thread-duty=4]
# [00000.238 ms] [16.695017] sumCuda<<<auto, 32>>> [thread-duty=6]
# [00000.189 ms] [16.695635] sumCuda<<<auto, 32>>> [thread-duty=8]
# [00000.139 ms] [16.695234] sumCuda<<<auto, 32>>> [thread-duty=12]
# [00000.123 ms] [16.695301] sumCuda<<<auto, 32>>> [thread-duty=16]
# [00000.109 ms] [16.695333] sumCuda<<<auto, 32>>> [thread-duty=24]
# [00000.101 ms] [16.695335] sumCuda<<<auto, 32>>> [thread-duty=32]
# [00000.093 ms] [16.695339] sumCuda<<<auto, 32>>> [thread-duty=48]
# [00000.091 ms] [16.695332] sumCuda<<<auto, 32>>> [thread-duty=64]
# [00000.299 ms] [16.695017] sumCuda<<<auto, 64>>> [thread-duty=1]
# [00000.296 ms] [16.695017] sumCuda<<<auto, 64>>> [thread-duty=2]
# [00000.238 ms] [16.695017] sumCuda<<<auto, 64>>> [thread-duty=3]
# [00000.189 ms] [16.695635] sumCuda<<<auto, 64>>> [thread-duty=4]
# [00000.142 ms] [16.695234] sumCuda<<<auto, 64>>> [thread-duty=6]
# [00000.123 ms] [16.695301] sumCuda<<<auto, 64>>> [thread-duty=8]
# [00000.108 ms] [16.695333] sumCuda<<<auto, 64>>> [thread-duty=12]
# [00000.100 ms] [16.695335] sumCuda<<<auto, 64>>> [thread-duty=16]
# [00000.093 ms] [16.695337] sumCuda<<<auto, 64>>> [thread-duty=24]
# [00000.089 ms] [16.695330] sumCuda<<<auto, 64>>> [thread-duty=32]
# [00000.086 ms] [16.695328] sumCuda<<<auto, 64>>> [thread-duty=48]
# [00000.085 ms] [16.695333] sumCuda<<<auto, 64>>> [thread-duty=64]
# [00000.307 ms] [16.695635] sumCuda<<<auto, 128>>> [thread-duty=1]
# [00000.197 ms] [16.695635] sumCuda<<<auto, 128>>> [thread-duty=2]
# [00000.149 ms] [16.695236] sumCuda<<<auto, 128>>> [thread-duty=3]
# [00000.124 ms] [16.695301] sumCuda<<<auto, 128>>> [thread-duty=4]
# [00000.109 ms] [16.695333] sumCuda<<<auto, 128>>> [thread-duty=6]
# [00000.099 ms] [16.695335] sumCuda<<<auto, 128>>> [thread-duty=8]
# [00000.093 ms] [16.695339] sumCuda<<<auto, 128>>> [thread-duty=12]
# [00000.089 ms] [16.695330] sumCuda<<<auto, 128>>> [thread-duty=16]
# [00000.085 ms] [16.695328] sumCuda<<<auto, 128>>> [thread-duty=24]
# [00000.085 ms] [16.695333] sumCuda<<<auto, 128>>> [thread-duty=32]
# [00000.083 ms] [16.695312] sumCuda<<<auto, 128>>> [thread-duty=48]
# [00000.078 ms] [16.695324] sumCuda<<<auto, 128>>> [thread-duty=64]
# [00000.261 ms] [16.695635] sumCuda<<<auto, 256>>> [thread-duty=1]
# [00000.152 ms] [16.695301] sumCuda<<<auto, 256>>> [thread-duty=2]
# [00000.126 ms] [16.695333] sumCuda<<<auto, 256>>> [thread-duty=3]
# [00000.100 ms] [16.695335] sumCuda<<<auto, 256>>> [thread-duty=4]
# [00000.092 ms] [16.695337] sumCuda<<<auto, 256>>> [thread-duty=6]
# [00000.089 ms] [16.695330] sumCuda<<<auto, 256>>> [thread-duty=8]
# [00000.086 ms] [16.695328] sumCuda<<<auto, 256>>> [thread-duty=12]
# [00000.085 ms] [16.695333] sumCuda<<<auto, 256>>> [thread-duty=16]
# [00000.082 ms] [16.695312] sumCuda<<<auto, 256>>> [thread-duty=24]
# [00000.079 ms] [16.695324] sumCuda<<<auto, 256>>> [thread-duty=32]
# [00000.082 ms] [16.695324] sumCuda<<<auto, 256>>> [thread-duty=48]
# [00000.079 ms] [16.695311] sumCuda<<<auto, 256>>> [thread-duty=64]
# [00000.232 ms] [16.695301] sumCuda<<<auto, 512>>> [thread-duty=1]
# [00000.142 ms] [16.695335] sumCuda<<<auto, 512>>> [thread-duty=2]
# [00000.114 ms] [16.695339] sumCuda<<<auto, 512>>> [thread-duty=3]
# [00000.089 ms] [16.695330] sumCuda<<<auto, 512>>> [thread-duty=4]
# [00000.082 ms] [16.695328] sumCuda<<<auto, 512>>> [thread-duty=6]
# [00000.083 ms] [16.695333] sumCuda<<<auto, 512>>> [thread-duty=8]
# [00000.084 ms] [16.695312] sumCuda<<<auto, 512>>> [thread-duty=12]
# [00000.081 ms] [16.695324] sumCuda<<<auto, 512>>> [thread-duty=16]
# [00000.080 ms] [16.695324] sumCuda<<<auto, 512>>> [thread-duty=24]
# [00000.079 ms] [16.695312] sumCuda<<<auto, 512>>> [thread-duty=32]
# [00000.080 ms] [16.695311] sumCuda<<<auto, 512>>> [thread-duty=48]
# [00000.080 ms] [16.695311] sumCuda<<<auto, 512>>> [thread-duty=64]
# [00000.248 ms] [16.695335] sumCuda<<<auto, 1024>>> [thread-duty=1]
# [00000.153 ms] [16.695330] sumCuda<<<auto, 1024>>> [thread-duty=2]
# [00000.121 ms] [16.695328] sumCuda<<<auto, 1024>>> [thread-duty=3]
# [00000.092 ms] [16.695333] sumCuda<<<auto, 1024>>> [thread-duty=4]
# [00000.081 ms] [16.695312] sumCuda<<<auto, 1024>>> [thread-duty=6]
# [00000.081 ms] [16.695324] sumCuda<<<auto, 1024>>> [thread-duty=8]
# [00000.080 ms] [16.695324] sumCuda<<<auto, 1024>>> [thread-duty=12]
# [00000.079 ms] [16.695312] sumCuda<<<auto, 1024>>> [thread-duty=16]
# [00000.081 ms] [16.695311] sumCuda<<<auto, 1024>>> [thread-duty=24]
# [00000.078 ms] [16.695312] sumCuda<<<auto, 1024>>> [thread-duty=32]
# [00000.079 ms] [16.695314] sumCuda<<<auto, 1024>>> [thread-duty=48]
# [00000.078 ms] [16.695311] sumCuda<<<auto, 1024>>> [thread-duty=64]
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

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[launch adjust]: https://github.com/puzzlef/sum-cuda-memcpy-adjust-launch
[in-place]: https://github.com/puzzlef/sum-cuda-inplace-adjust-launch
