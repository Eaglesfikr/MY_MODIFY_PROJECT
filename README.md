# T-DCWF

这是我的2025研究生课程网络测量的大作业



## 项目结构：

```sh
 ├─datasets #数据集文件位置
 ├─T-DCWF   #项目代码与日志
    ├─data	#结果日志文件
    │  ├─DCWF #原来的
    │  └─Ours #改进后的
    ├─pre_trained_model #预训练参数文件位置
    └─util	#方法函数
```



## 数据集：

百度网盘：https://pan.baidu.com/s/1vbD83AIHgaMeSAXLwenALg?pwd=2qzu 
提取码：2qzu

**注**：对原方法的复现，这里的数据集用的是我自己合成的npz文件，作者已经不跟原院校联系，故本人自己使用AWF仓库的775子文件夹合成的，所以是非官方实现：https://github.com/chenxiang3luo/DCWF/issues/2，不承担任何责任。

下载下来的datasets请放到与T-DCWF统一目录下



预训练模型也由网盘提供，使用跟DCWF相同的：https://pan.baidu.com/s/1d8NTTk_mDXYLunLbkvJewQ?pwd=2qzu 
提取码：2qzu



## 运行：

修改main.py的最后的源域数据集和类别数，再运行一下命令即可。

```
python main.py
```

