之前的尝试都是对epoch，batch size，网络结构，optimizer，lr的改动，效果都有限。并且经过可视化验证集loss始终大幅震荡，好像无法收敛。于是对数据集进行可视化，看看数据预处理阶段有什么需要改进。<br>

<center class="half">
    <img src="https://github.com/zwkkk/pytorch-project-exercise/blob/master/PIC/original.png" width="200"/><img src="https://github.com/zwkkk/pytorch-project-exercise/blob/master/PIC/baseline.png" width="200"/>
</center>
左图为原始图像，右图为baseline给的预处理方式。<br>

???????破案了???????<br>
这个切分方式细胞样子都快没了<br>

现对它做如下修改：<br>

```python
image = Image.open(path).convert('L')
        image = image.rotate(45)    #针对原始图像，旋转45度尽可能保留非0数据
        x_ = numpy.array(image)
        line, columns = numpy.nonzero(x_)     #选择非0区域
        lmin = min(line)  # 有非0像素的最小行
        lmax = max(line)  # 有非0像素的最大行
        colmin = min(columns) # 有非0像素的最小列
        colmax = max(columns)  # 有非0像素的最大列
        image = Image.fromarray(x_)
        image = image.crop((colmin, lmin, colmax, lmax))

        image = image.resize((224, 224))     #确定尺寸
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = x_data.reshape([224, 224])
 ```
 
<img src='https://github.com/zwkkk/pytorch-project-exercise/blob/master/PIC/change.png'/>
修改后图像<br>

Epoch：2200<br>
Batch size:256<br>
评分：94.56<br>
<img src='https://github.com/zwkkk/pytorch-project-exercise/blob/master/PIC/resnet18-94.56.png'/>
