import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
import datetime

# 设置数据集目录
data_dir = '../image'
Time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f'CNN/CNN_{Time}.h5'

# 获取数据集中的所有图像路径和标签
image_paths = []
labels = []

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只添加特定类型的文件
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                labels.append(filename.split('_')[0])  # 使用第一个 '_' 前的字符串作为标签

# 将标签转换为数字编码
label_to_index = {label: i for i, label in enumerate(np.unique(labels))}
labels = [label_to_index[label] for label in labels]

# 加载图像数据并将其转换为 NumPy 数组
images = np.array([img_to_array(load_img(img_path, target_size=(224, 224))) for img_path in image_paths])

# 数据标准化
images = images / 255.0

# # 使用 ImageDataGenerator 进行数据增强
# train_datagen = ImageDataGenerator(
#     rotation_range=5,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     fill_mode='nearest'
# )

# 划分训练集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
batch_size = 32
# 构建 train_dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)

# 构建验证数据集
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# # 构建 train_dataset
# train_dataset = train_datagen.flow(train_images, train_labels, batch_size=32)
#
# # 构建验证数据集
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# 构建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.5),  # 添加Dropout层
    layers.Dense(len(label_to_index), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 定义 TensorBoard 日志目录
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 配置 TensorBoard 回调
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 定义早停法回调
early_stopping = EarlyStopping(monitor='val_loss',  # 监控验证集上的损失值
                               patience=5,         # 如果连续10个epoch验证集损失没有改善，则停止训练
                               mode='min',
                               restore_best_weights=True)  # 在停止训练时恢复最佳模型参数

# 训练模型
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset,
                    callbacks=[tensorboard_callback, early_stopping])

# 保存模型
model.save(model_path)
