import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# Flatten y về dạng (N,)
y_train, y_test = y_train.flatten(), y_test.flatten()

# Normalize dữ liệu về [0,1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)

# 2. Xây dựng mô hình Deep Learning
def build_model(input_shape=(32,32,3), num_classes=100):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model()

# Compile (sparse labels)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc")])

model.summary()

# 3. Huấn luyện
history = model.fit(x_train, y_train,
                    validation_split=0.1,
                    epochs=30,
                    batch_size=64)

# 4. Đánh giá trên TestData
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (CIFAR-100)")
plt.show()

# 5. Các chỉ số đánh giá
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Top-5 accuracy
top5_acc = top_k_accuracy_score(y_test, y_pred_prob, k=5, labels=np.arange(100))

print("Accuracy :", acc)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("Top-5 Acc:", top5_acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
