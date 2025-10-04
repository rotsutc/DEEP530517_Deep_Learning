import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
import numpy as np
import pandas as pd

# ============================================================
# 1. Tải dữ liệu CIFAR-100
# ============================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

# Chuẩn hóa dữ liệu
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding cho nhãn
num_classes = 100
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# Danh sách tên lớp trong CIFAR-100
fine_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
class_names = fine_labels

# ============================================================
# 2. Xây dựng mô hình CNN
# ============================================================
model = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# ============================================================
# 3. Compile mô hình
# ============================================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================================
# 4. Huấn luyện mô hình
# ============================================================
history = model.fit(x_train, y_train_cat,
                    epochs=10,             # CIFAR-100 phức tạp hơn, nên tăng epoch
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# ============================================================
# 5. Đánh giá trên test set
# ============================================================
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print("\n🎯 Test accuracy:", test_acc)


# ============================================================
# Hàm hiển thị toàn màn hình
# ============================================================
def show_maximized():
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')
    except AttributeError:
        try:
            mng.window.showMaximized()
        except AttributeError:
            try:
                mng.resize(*mng.window.maxsize())
            except Exception:
                pass
    plt.show()


# ============================================================
# 6. Biểu đồ Loss & Accuracy
# ============================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss per Epoch (CIFAR-100)")
plt.xlabel("Epoch"); plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy per Epoch (CIFAR-100)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
show_maximized()


# ============================================================
# 7. Dự đoán thử 1 ảnh từ test set
# ============================================================
idx = np.random.randint(0, len(x_test))
img = x_test[idx]
true_label = y_test[idx][0]
pred_probs = model.predict(img.reshape(1,32,32,3))
pred_label = np.argmax(pred_probs)

plt.imshow(img)
plt.axis("off")
plt.title(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}")
show_maximized()


# ============================================================
# 8. Hiển thị 10 ảnh test đầu tiên
# ============================================================
plt.figure(figsize=(15,6))
for i in range(10):
    img = x_test[i]
    true_label = y_test[i][0]
    pred_probs = model.predict(img.reshape(1,32,32,3), verbose=0)
    pred_label = np.argmax(pred_probs)
    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.axis("off")
    color = "green" if pred_label == true_label else "red"
    plt.title(f"T:{class_names[true_label]}\nP:{class_names[pred_label]}", color=color)
plt.suptitle("10 ảnh test đầu tiên: CIFAR-100", fontsize=14)
show_maximized()


# ============================================================
# 9. Confusion Matrix & Classification Report
# ============================================================
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12,10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Confusion Matrix - CIFAR-100", fontsize=14)
plt.tight_layout()
show_maximized()

# Báo cáo chi tiết
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print("📊 Classification Report:\n")
print(report)

# ============================================================
# 10. Metrics tổng quát
# ============================================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")
rec = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
precision_at5 = top_k_accuracy_score(y_true, y_pred_probs, k=5, labels=range(num_classes))
f1_at5 = precision_at5

print("\n📊 Metrics (macro-average):")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"F1@Top-5   : {f1_at5:.4f}")

# ============================================================
# 11. Vẽ biểu đồ Precision/Recall/F1 cho 20 lớp đầu
# ============================================================
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose().iloc[:20, :3]  # chỉ 20 lớp đầu
df_report = df_report.round(3)

print("\n📊 Bảng per-class Precision / Recall / F1 (20 lớp đầu):\n")
print(df_report)

df_report.plot(kind="bar", figsize=(14,6))
plt.title("Per-class Precision / Recall / F1 - CIFAR-100 (20 lớp đầu)")
plt.xlabel("Class"); plt.ylabel("Score")
plt.ylim(0,1)
plt.tight_layout()
show_maximized()
