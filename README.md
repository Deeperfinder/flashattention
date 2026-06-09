# Flash Attention实现
包含如下两个实现文件：
* blackwell:  5090的flash attention cutlass优化版本
    * persistent kernel
    * warp specialization
    * tma load/store
    * double buffering  
* src: 纯ptx手写的5090 flash attn实现
    - ldmatrix.x4
    - mma.m16n8k16
    - cp.async.cg.shared.global
    - 支持head_dim=128
    - Flash Attention v2 算法

## examples
- 提供了`sm120_fa.py`测试简本，可以测试flash attention V2, V3, V4 / SPDA 与目前仓库的优化版本在5090上的性能表现。
- 这里flash attentio v3基于sm80的fwd实现，在hopper文件夹中修改`setup.py`中的编译方法，添加sm120的编译选项， 代码仓库地址：`https://code.alibaba-inc.com/Easyhpc/Flash_attention_cp/tree/fa3_5090/`   
- 使用方法：
```bash
# 需要提前安装好fa2，fa3，fa4的版本
python sm120_fa.py
```
```bash
# res
python sm120_fa.py 
==================================================
Benchmark: B=1, H=5, Lq=144169, Lk=144169, D=128, Causal=False
Precision: float16
--------------------------------------------------
FlashAttn2     :  245.604 ms |   216.64 TFLOPS
FlashAttn3     :  231.191 ms |   230.15 TFLOPS
cuDNN          :  231.716 ms |   229.63 TFLOPS
FA_5090        :  245.000 ms |   217.18 TFLOPS
SDPA           :  252.244 ms |   210.94 TFLOPS
--------------------------------------------------
Mean Absolute Error vs FlashAttn2: 3.576279e-07
Max Absolute Error vs FlashAttn2:  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs FlashAttn3: 3.576279e-07
Max Absolute Error vs FlashAttn3:  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs cuDNN: 1.192093e-06
Max Absolute Error vs cuDNN:  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs SDPA: 3.576279e-07
Max Absolute Error vs SDPA:  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs SDPA (FA2): 0.000000e+00
Max Absolute Error vs SDPA (FA2):  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs SDPA (FA3): 5.960464e-08
Max Absolute Error vs SDPA (FA3):  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs SDPA (cuDNN): 1.192093e-06
Max Absolute Error vs SDPA (cuDNN):  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs FlashAttn3: 5.960464e-08
Max Absolute Error vs FlashAttn3:  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs cuDNN: 1.192093e-06
Max Absolute Error vs cuDNN:  1.525879e-05
--------------------------------------------------
Mean Absolute Error vs cuDNN: 1.192093e-06
Max Absolute Error vs cuDNN:  1.525879e-05
--------------------------------------------------
Status: PASS
==================================================
```
## pipeline
![alt text](./images/pipeline.png)
