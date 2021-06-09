#!/usr/bin/env bash
src="sum-sequential-vs-openmp"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/puzzlef/$src
cd $src

# Run
run() {
g++ -O3 -fopenmp main.cxx
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" | tee -a "$out"
stdbuf --output=L ./a.out 2>&1          | tee -a "$out"
echo ""                                 | tee -a "$out"
}

export OMP_NUM_THREADS=2  && run
export OMP_NUM_THREADS=4  && run
export OMP_NUM_THREADS=8  && run
export OMP_NUM_THREADS=16 && run
export OMP_NUM_THREADS=28 && run
export OMP_NUM_THREADS=32 && run
export OMP_NUM_THREADS=48 && run
