本次采用resnet18网络进行分类任务，<br>

相较baseline改动:<br>
1. 将net.py改为resnet18，
2. processor.py图像尺寸改为（224，224）以适应resnet尺寸。<br>
epoch：10 batch_size:32<br>
评分：34.22
