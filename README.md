实验二：
实验测得的数据都在对应project的工作簿里，unroll展开和多流水线超标量等优化的代码是直接从原代码上直接改的，没有单独成一个文件。
linux系统上跑的代码跟kunpeng的代码类似，都是用的通用的时钟进行计数。
python文件是用来生成矩阵or数组元素的文件，用来测量文件大小，以此根据缓存来决定测试规模N。

鲲鹏服务器：
![image](https://github.com/lhz191/bingxing/assets/142021438/ff2fd4f6-57d0-4cc6-89a5-f6f899c15213)
![image](https://github.com/lhz191/bingxing/assets/142021438/3c542e23-11d1-4789-98ee-b8a6c76ecba1)
![image](https://github.com/lhz191/bingxing/assets/142021438/ee08d75c-24a4-4f4b-80a1-2a39b03a63f0)

linux:在实验室的电脑上用vscode的远程服务器连接跑的代码，实验室电脑和个人amd电脑的配置专门用benchmark测的，可以保证正确性。图中也可以看到perf的记录和证明。

![image](https://github.com/lhz191/bingxing/assets/142021438/5e38a567-54ff-4f1c-b477-1d93df60cb8f)

利用SIMD编程实现K-means算法优化。


openmp：利用Intel Developer Cloud平台将OpenMP卸载到加速设备和进行oneapi实验
![ebce4226a32823ddcfeb0beea0f1053](https://github.com/lhz191/bingxing/assets/142021438/57939493-fecd-4d43-954f-c83f356cc458)
