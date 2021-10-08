# https://www.kaggle.com/wolfram77/puzzlef-sum-cuda-memcpy-adjust-launch
import os
from IPython.display import FileLink
src="sum-cuda-memcpy-adjust-launch"
out="{}.txt".format(src)
!printf "" > "$out"
display(FileLink(out))
!ulimit -s unlimited && echo ""
!nvidia-smi && echo ""

# Download program
!rm -rf $src
!git clone https://github.com/puzzlef/$src
!echo ""

# Run
!nvcc -std=c++17 -Xcompiler -O3 $src/main.cu
!stdbuf --output=L ./a.out 2>&1 | tee -a "$out"
