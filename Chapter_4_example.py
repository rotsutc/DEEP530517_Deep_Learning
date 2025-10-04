""" 
Import các thư viện cần thiết:
    tensorflow/keras: framework deep learning.
    layers: các lớp mạng (Dense, Dropout, …).
    models: để tạo và quản lý mô hình.
    optimizers: chứa các bộ tối ưu (Adam, SGD,…).
    applications: chứa các mô hình pretrained như MobileNet, ResNet, VGG,… 
"""
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers as opt, applications as app

"""
Tải mô hình MobileNet đã huấn luyện trước trên ImageNet.
    include_top=False → bỏ đi phần fully-connected cuối cùng (chỉ lấy phần trích xuất đặc trưng).
    weights="imagenet" → dùng trọng số huấn luyện sẵn từ ImageNet.
"""
pre_trained_model = app.MobileNet(include_top=False, weights="imagenet")

# In cấu trúc mô hình MobileNet để xem các layer và số tham số
pre_trained_model.summary()

# Lấy output của layer cuối cùng (trong MobileNet, sau khi bỏ top)
last_layer = pre_trained_model.get_layer(index=-1)
last_output = last_layer.output

# Thay vì flatten, dùng Global Max Pooling để rút gọn tensor thành vector đặc trưng
x = layers.GlobalMaxPooling2D()(last_output)

""" 
Thêm các lớp fully-connected:
    Dense(512, relu): lớp ẩn với 512 neurons.
    Dropout(0.5): ngẫu nhiên tắt 50% neurons để tránh overfitting.
    Dense(8, softmax): lớp đầu ra 8 lớp (multi-class classification)
 """
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(8, activation='softmax')(x)

""" 
Tạo mô hình mới:
    Input: ảnh đầu vào của MobileNet.
    Output: lớp softmax 8 lớp vừa thêm
"""
model = models.Model(pre_trained_model.input, x)

# In ra số lượng layer và số tham số
print(f"Model has {len(model.layers)} layers, params {model.count_params()};")


# Chọn optimizer Adam với learning rate nhỏ 1𝑒-4
optimizer = opt.Adam(learning_rate=0.0001)

""" 
Compile mô hình:
    loss='categorical_crossentropy': dùng cho multi-class classification.
    optimizer=Adam: phương pháp tối ưu.
    metrics="acc": đánh giá bằng accuracy.
"""

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

