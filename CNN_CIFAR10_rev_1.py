import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
import numpy as np
import pandas as pd

# 1. Tải dữ liệu CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Chuẩn hóa dữ liệu
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding cho nhãn
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# Danh sách tên lớp trong CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# 2. Xây dựng mô hình CNN với 5 hidden layers
model = models.Sequential([
    layers.Input(shape=(32,32,3)),

    # Hidden layer 1
    layers.Conv2D(32, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D((2,2)),

    # Hidden layer 2
    layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D((2,2)),

    # Hidden layer 3
    layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D((2,2)),

    # Hidden layer 4
    layers.Conv2D(128, (3,3), activation='relu', padding="same"),

    # Flatten sang vector
    layers.Flatten(),

    # Hidden layer 5 (fully connected)
    layers.Dense(128, activation='relu'),

    # Output layer
    layers.Dense(num_classes, activation='softmax')
])


# 3. Compile mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Tóm tắt kiến trúc
model.summary()

# 4. Huấn luyện mô hình
history = model.fit(x_train, y_train_cat,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# 5. Đánh giá trên test set
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print("\n🎯 Test accuracy:", test_acc)


def show_maximized():
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # Windows
    except AttributeError:
        try:
            mng.window.showMaximized()  # Qt backend
        except AttributeError:
            try:
                mng.resize(*mng.window.maxsize())  # TkAgg / Linux
            except Exception:
                pass
    plt.show()
    
# 6. Vẽ biểu đồ Loss và Accuracy
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss per Epoch")
plt.xlabel("Epoch")         # ✅ thêm nhãn trục X
plt.ylabel("Loss")          # ✅ thêm nhãn trục Y

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")         # ✅ thêm nhãn trục X
plt.ylabel("Accuracy")      # ✅ thêm nhãn trục Y

show_maximized() 


# 7. Dự đoán thử 1 ảnh từ test set
idx = np.random.randint(0, len(x_test))  # chọn ngẫu nhiên 1 ảnh
img = x_test[idx]
true_label = y_test[idx][0]

# Dự đoán
pred_probs = model.predict(img.reshape(1,32,32,3))
pred_label = np.argmax(pred_probs)

# Hiển thị kết quả
plt.imshow(img)
plt.axis("off")
plt.title(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}")
show_maximized() 

# 8. Hiển thị 10 ảnh test đầu tiên kèm dự đoán
plt.figure(figsize=(15, 6))

for i in range(10):
    img = x_test[i]
    true_label = y_test[i][0]
    
    # Dự đoán
    pred_probs = model.predict(img.reshape(1,32,32,3), verbose=0)
    pred_label = np.argmax(pred_probs)
    
    # Vẽ ảnh
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.axis("off")
    color = "green" if pred_label == true_label else "red"
    plt.title(f"T:{class_names[true_label]}\nP:{class_names[pred_label]}", color=color)

plt.suptitle("10 ảnh test đầu tiên: T = True, P = Predicted", fontsize=14)
show_maximized() 

# 9. Tính nhãn dự đoán cho toàn bộ test set
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

# 10. Vẽ confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Vẽ confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d", ax=ax)

# Thêm tiêu đề và nhãn trục
ax.set_title("Confusion Matrix - CIFAR-10", fontsize=14)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)

# Điều chỉnh layout 
plt.subplots_adjust(
    left=0.125,
    bottom=0.145,
    right=0.9,
    top=0.88,
    wspace=0.2,
    hspace=0.2
)
show_maximized() 

# 11. Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("📊 Classification Report:\n")
print(report)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score

# 12. Tính các metrics tổng quát
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")   # macro: trung bình đều giữa các lớp
rec = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

# F1@Top5: dùng top-k accuracy để xem nhãn thật có nằm trong top-5 dự đoán hay không
# Ở đây precision@5 = recall@5 nên F1@5 cũng bằng giá trị đó
precision_at5 = top_k_accuracy_score(y_true, y_pred_probs, k=5, labels=range(num_classes))
recall_at5 = precision_at5
f1_at5 = 2 * (precision_at5 * recall_at5) / (precision_at5 + recall_at5)

print("\n📊 Metrics (macro-average):")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"F1@Top-5   : {f1_at5:.4f}")

import pandas as pd

# 13. Lấy classification report dạng dict
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Chuyển thành DataFrame (chỉ lấy 10 lớp)
df_report = pd.DataFrame(report_dict).transpose().iloc[:10, :3]  # chỉ lấy precision, recall, f1-score
df_report = df_report.round(3)  # làm gọn số thập phân

print("\n📊 Bảng per-class Precision / Recall / F1:\n")
print(df_report)

# 14. Vẽ biểu đồ cột
df_report.plot(kind="bar", figsize=(12,6))
plt.title("Per-class Precision / Recall / F1 - CIFAR-10", fontsize=14)
plt.xlabel("Class")
plt.ylabel("Score")
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.tight_layout()
show_maximized()
