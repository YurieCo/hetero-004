# Heterogeneous Parallel Programming

### Matrix Mulitplication

| Method | Matrix Size | Time |
|:-------:|:------:|:-----:|
| [Naive](Basic_Matrix_Multiplication.cu)  | 200*256 | 0.2515 ms |
| [Tiled](Tiled_Matrix_Multiplication.cu)  | 200*256 | 0.1127 ms |


#### Image Convolution
| Method | Image Size | Time | Speed Up (x) |
|:-------:|:------:|:-------:|:------------:|
| CPU        | 64 x 64 x 3  | 0.39 ms | 1.00 |
| GPU Tailed | 64 x 64 x 3  | 0.06 ms | 6.50 |
| CPU        | 400 x 400 x 3  | 15.03 ms | 1.00 |
| GPU Tailed | 400 x 400 x 3  | 0.72 ms | 20.88 |
| CPU        | 2048 x 2048 x 3  | 408.72 ms | 1.00 |
| GPU Tailed | 2048 x 2048 x 3  | 17.85 ms | 22.90 |

### Histogram Normalization
Note : I/O time is not included
[CPU Solution](Histogram_CPU.cu)
[GPU Solution](Histogram_GPU_v2.cu)
| Method | Image Size | Hist (ms) | Corr. (ms) | Total (ms) | Speed Up (x) |
|:------:|:----------:|:---------:|:----------:|:----------:|:------------:|
| CPU    | 256 * 256  | 0.257     | 0.730      | 0.987      | 1.00         |
| GPU    | 256 * 256  | 0.140     | 0.081      | 0.221      | 4.47         |
| CPU    | 1024*683   | 2.809     | 7.806      | 7.806      | 1.00         |
| GPU    | 1024*683   | 0.269     | 0.637      | 0.906      | 11.72        |

