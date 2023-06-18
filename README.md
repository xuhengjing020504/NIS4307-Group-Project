# NIS4307-Group-Project

## 接口调用程序说明
- 接口文件为AiGcMn.py

### 导入依赖库

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cDCGAN import Generator
```



### 类 AiGcMn
#### 构造函数 __init__(self, generator_path)
初始化GAN模型接口的实例。

- generator_path：生成器模型的路径。
- device：设备类型，根据CUDA是否可用自动选择在GPU上运行或在CPU上运行。
- generator：生成器模型的实例。
- transform：用于图像转换的组合。

####  generate_images(self, num_images, noise_dim=100, class_labels=[0])
生成指定数量的图像。
- num_images：要生成的图像数量。
- noise_dim：噪声向量的维度，默认为100。
- class_labels：图像的类别标签列表，默认为[0]。类别标签数量应与生成的图像数量相匹配。
返回生成的图像列表，类型为PIL图像。

## 调用示例
```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from AiGcMn import AiGcMn

generator_path = 'Generator.pth'
aigcmn = AiGcMn(generator_path)

# Generate 10 images with different class labels
class_labels = [1, 5, 2, 9, 4, 7, 0, 3, 8, 6]
generated_images = aigcmn.generate_images(num_images=10, class_labels=class_labels)

# Display the generated images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated_images[i],cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()
```
代码与输出示例请见[example]:https://github.com/xuhengjing020504/NIS4307-Group-Project/blob/main/example.ipynb


## 模型效果展示
- Epoch = 30时的训练过程


![](https://github.com/xuhengjing020504/NIS4307-Group-Project/blob/main/sample.gif)

