# 🖊️ MNIST CNN Classifier

مرحبًا بك في هذا المشروع الصغير والممتع! 🎉  
هنا ستتعلم كيف يمكن لشبكة عصبية **تلافيفية (CNN)** التعرف على **الأرقام المكتوبة يدويًا** باستخدام مكتبة **TensorFlow/Keras**.  

---

## 📌 الفكرة
المشروع يستخدم مجموعة بيانات **MNIST** الشهيرة، التي تحتوي على صور رمادية لأرقام من 0 إلى 9.  
الهدف: بناء نموذج يستطيع قراءة الرقم الموجود في الصورة بدقة عالية، كما لو كان عين بشرية 👀.

---

## ⚡️ مميزات النموذج
- نموذج CNN بسيط وسريع للتدريب.
- يستخدم **طبقات Convolution + MaxPooling + Dense**.
- يدعم **تصنيف 10 أرقام**.
- يعطي **توقع لكل صورة فردية** مع عرض الصورة نفسها.

---

## 🛠️ متطلبات التشغيل
- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib

يمكن تثبيتها بسهولة عبر:
```bash
pip install -r requirements.txt
```

## 🚀 طريقة التشغيل

تأكد أن لديك Python وpip مثبتين.

نفذ الأمر التالي لتثبيت المكتبات:

pip install tensorflow matplotlib numpy


شغّل ملف الكود:

python mnist_cnn.py


ستظهر نافذة تعرض صورة الرقم وتوقع النموذج.

## 🎯 خطوات العمل في الكود

تحميل بيانات MNIST وتقسيمها إلى تدريب واختبار.

تطبيع القيم لتكون بين 0 و1.

بناء نموذج CNN:

Conv2D → MaxPooling → Flatten → Dense → Output

تدريب النموذج على بيانات التدريب.

تقييم الأداء على بيانات الاختبار.

تجربة النموذج على صورة واحدة وعرضها مع التوقع.

## 📊 مثال للنتائج

دقة النموذج على بيانات الاختبار: ~98% 🎉
<img width="677" height="735" alt="image" src="https://github.com/user-attachments/assets/3eea953a-8354-4e90-8308-ba595d12dba3" />


النموذج قادر على التمييز بين جميع الأرقام اليدوية.

## 💡 نصيحة

يمكنك تعديل عدد الطبقات أو الوحدات العصبية لتحسين الدقة، أو تجربة Augmentation لزيادة حجم البيانات وتحسين الأداء أكثر!

🧠 مراجع علمية

### 📚 المراجع

1. **مجموعة بيانات MNIST**  
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

2. **دليل TensorFlow الرسمي لبناء CNN**  
[https://www.tensorflow.org/tutorials/images/cnn](https://www.tensorflow.org/tutorials/images/cnn)

3. **مقالة عن أنواع الـ Convolutions في التعلم العميق**  
[https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

4. **كتاب Deep Learning – Ian Goodfellow**  
[https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

5. **توثيق Keras الرسمي للنماذج المتسلسلة**  
[https://keras.io/guides/sequential_model/](https://keras.io/guides/sequential_model/)

