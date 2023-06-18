# NIS4307-Group-Project

## 接口调用程序说明
- 接口文件为AiGcMn.py
该接口封装了一个生成对抗网络（GAN）模型，用于生成图像。以下是该接口的使用说明和功能描述。

导入依赖库
python
Copy code
<pre><code>import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cDCGAN import Generator
</code></pre>

接口使用了以下依赖库：

torch：PyTorch深度学习库。
torch.nn：PyTorch的神经网络模块。
torchvision.transforms：用于图像转换的模块。
matplotlib.pyplot：用于图像可视化的库。
cDCGAN：自定义的生成器模型。
类 AiGcMn
构造函数 __init__(self, generator_path)
初始化GAN模型接口的实例。

参数
generator_path：生成器模型的路径。
成员变量
device：设备类型，根据CUDA是否可用自动选择在GPU上运行或在CPU上运行。
generator：生成器模型的实例。
transform：用于图像转换的组合。
方法 generate_images(self, num_images, noise_dim=100, class_labels=[0])
生成指定数量的图像。

参数
num_images：要生成的图像数量。
noise_dim：噪声向量的维度，默认为100。
class_labels：图像的类别标签列表，默认为[0]。类别标签数量应与生成的图像数量相匹配。
返回值
返回生成的图像列表，类型为PIL图像。




## 模型效果展示
- Epoch = 30时的训练过程


![](https://github.com/xuhengjing020504/NIS4307-Group-Project/blob/main/sample.gif)

