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

