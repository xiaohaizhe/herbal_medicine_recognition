1. pics_augmetation.py  
- 使用PIL库，对目标目录下的图片文件进行翻转、旋转，生成6倍数量的图片，实现数据集扩充。
2. files2label.py  
- 按目录-文件结构生成标签文件；
- 准备数据集，按比例分配train/test集，生成对应标签文件，并拷贝文件至指定目录。  
3. inception_v3.py inception_utils.py
- 采用tensorflow.contrib.slim定义GoogLeNet的图结构
4. plant_g.py
- 模型训练的主代码，
    - 训练模式：训练模型、记录训练过程数据、保存高acc模型；
    - 测试模式：生成测试集识别结果文件。