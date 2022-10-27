# https://www.kaggle.com/wolfram77/puzzlef-sum-sequential-vs-cuda
import os
from IPython.display import FileLink
src="sum-sequential-vs-cuda"
out="{}.txt".format(src)
!printf "" > "$out"
display(FileLink(out))
!echo ""

# Download program
!rm -rf $src
!git clone https://github.com/puzzlef/$src
!echo ""

# Run
!nvcc -std=c++17 -Xcompiler -O3 main.cu
!ulimit -s unlimited && stdbuf --output=L ./a.out 2>&1 | tee -a "$out"
