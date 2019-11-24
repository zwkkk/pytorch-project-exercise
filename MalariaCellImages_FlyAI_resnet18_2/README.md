# 对优化器进行调参:<br>

optimizer = Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer. <br>

1. 
epoch:10<br>
batch_size:32<br>
评分：32.45<br>

2. 
epoch:128<br>
batch_size:500<br>
评分：75.41<br>

3. 
epoch:128<br>
batch_size:1000<br>
评分：50.97<br>

4. 
epoch:512<br>
batch_size:1000<br>
评分：85.93<br>

5. 
epoch:512<br>
batch_size:128<br>
评分：82.29<br>

6. 
epoch:1000<br>
batch_size:128<br>
评分：85.91<br>

7. 
epoch:1000<br>
batch_size:256<br>
评分：86.73<br>

optimizer = Adam(cnn.parameters(), lr=3e-4)  # 选用AdamOptimizer <br>
8. 
epoch:1000<br>
batch_size:512<br>
评分：84.31<br>

500/1000 <br>
tensor(0.0745, device='cuda:0')<br>

700/1000<br>
tensor(0.0565, device='cuda:0')<br>

800/1000<br>
tensor(0.0193, device='cuda:0')<br>

900/1000<br>
tensor(0.0051, device='cuda:0')<br>

998/1000<br>
tensor(0.0167, device='cuda:0')<br>

```python
#调整学习率策略 factor=0.85, patience=20 要搭配好，factor太小容易一下子就使lr降到很小值

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=20, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
```
9. 
epoch:500<br>
batch_size:512<br>
评分：49.6<br>
train_loss: 0.6935; test_accuracy: 0.71875; lr: 7.1410e-06=0.017 <br>
分析：训练损失还很大，因此为欠拟合，要增大训练轮数；factor=0.85, patience=20，factor可能设置过大，导致后期lr降不下来; 模型保存逻辑为测试集accuracy不再增加，则保存模型，那么当测试集acc=1时，就不再保存模型了，之后我们将保存逻辑调整为当测试集accuracy大于等于之前最好，就保存模型，使训练集进一步拟合。<br>


```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=50, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
```
10.
epoch:1000<br>
batch_size:512<br>
评分：<br>版本42

11.
epoch:2000<br>
batch_size:256<br>
评分：<br>版本43

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=100, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
```
12.
epoch:2000<br>
batch_size:256<br>
评分：<br>版本44


参考第8次训练过程，针对patience进行处理<br>
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=800, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
```                                           
13.
epoch:4000<br>
batch_size:256<br>
评分：<br>版本45

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=800, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
```
14.
epoch:8000<br>
batch_size:256<br>
评分：<br>版本46

将在测试集的损失作为保存模型的标准，原来为准确率<br>
epoch:500<br>
batch_size:512<br>
评分：<br>版本41
