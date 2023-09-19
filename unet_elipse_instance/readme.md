## U-Net 对细胞核进行分割以及分类

### 1、下载预训练模型

链接: https://pan.baidu.com/s/15t_L7n4FH_3iqgJswf5i1w 提取码: u8zc 

### 2、更改detect_images.py文件

```python
model_path="your model saved path"
image_dirs="your cell images dir path"
```

### 3、执行预测

```
python detect_images.py
```

|                  细胞图像                  |                  分割结果                  |
| :----------------------------------------: | :----------------------------------------: |
| ![1571280789295](assets/1571280789295.png) | ![1571280899347](assets/1571280899347.png) |

