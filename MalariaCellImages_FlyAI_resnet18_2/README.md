# 对优化器进行调参:<br>

optimizer = Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer. <br>
epoch:10<br>
batch_size:32<br>
评分：32.45<br>

epoch:128<br>
batch_size:500<br>
评分：75.41<br>

epoch:128<br>
batch_size:1000<br>
评分：50.97<br>

epoch:512<br>
batch_size:1000<br>
评分：85.93<br>

epoch:512<br>
batch_size:128<br>
评分：82.29<br>

epoch:1000<br>
batch_size:128<br>
评分：85.91<br>

epoch:1000<br>
batch_size:256<br>
评分：86.73<br>

optimizer = Adam(cnn.parameters(), lr=3e-4)  # 选用AdamOptimizer <br>
epoch:1000<br>
batch_size:512<br>
评分：84.31<br>

```python
#调整学习率策略
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
```
epoch:1000<br>
batch_size:512<br>
评分：<br>版本32

epoch:1000<br>
batch_size:128<br>
评分：<br>版本33

epoch:1000<br>
batch_size:256<br>
评分：<br>版本33

epoch:512<br>
batch_size:1000<br>
评分：<br>版本34

epoch:512<br>
batch_size:1000<br>
评分：<br>版本35

将在测试集的损失作为保存模型的标准，原来为准确率<br>
epoch:1000<br>
batch_size:512<br>
评分：<br>版本36
