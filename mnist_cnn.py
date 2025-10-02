import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# تحميل بيانات MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # تطبيع القيم

# إضافة بُعد للقناة (قناة رمادية 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# بناء نموذج CNN بسيط
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# تدريب النموذج
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# تقييم النموذج
loss, acc = model.evaluate(x_test, y_test)
print(f"\n✅ دقة النموذج على بيانات الاختبار: {acc*100:.2f}%")

# اختبار على صورة واحدة
idx = 0  # اختر أي صورة من بيانات الاختبار (0 تعني أول صورة)
image = x_test[idx]
label = y_test[idx]

# عمل توقع
pred = model.predict(image[np.newaxis, ...])
pred_label = np.argmax(pred)

# عرض الصورة + النتيجة
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real: {label} | Model prediction: {pred_label}")
plt.axis("off")
plt.show()
