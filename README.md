# NCMM (Nsight Compute Matrix Multiplication)

Code of a matrix multiplication AxB of NxN each one to analize on Nsight Compute (for simplicity, N is a multiple of 32).

To use Nsight Compute on FT3:
~~~shell
module load cuda/12.2.0
~~~

## Codes

- MM.cu: original matrix multiplication
- MMo.cu: matrix multiplication with transpose of matrix B (increase L1 hit rate, decrease coalescence).
- MMo1.cu: matrix multiplication with one row of A in shared memory per NxN/cb blocks, being cb the CUDA block size.
- MMo2.cu: the same as MMo1 but with coalesced accesses and no bank conflicts

## Build 

~~~shell
make <nameofcode>
ncu --config-file on <nameofcode> <N> <cb>
~~~

You can execute multiple profilings on FT3 A100 with the following sbatch script:
~~~shell
sbatch ncu.sbatch
~~~
