import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# إيقاف استخدام الـ GPU إذا كنت ترغب في التشغيل على CPU فقط
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# إعداد عنوان الصفحة والمظهر
st.set_page_config(page_title="Lung Cancer Classification", layout="centered")

# تحميل النموذج مرة واحدة فقط لتفادي إعادة التحميل عند كل تفاعل (Caching)
@st.cache_resource
def get_model():
    # تأكد من تعديل المسار ليتناسب مع موقع ملف النموذج لديك
    model_path = r'model_vgg16_layer.h5'
    model = load_model(model_path)
    return model

model = get_model()

# دالة التنبؤ بنوع الصورة
def model_predict(img, model):
    # تغيير حجم الصورة لتناسب مدخلات نموذج VGG16 (224x224)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    preds = model.predict(img_array)
    pred = np.argmax(preds, axis=1)
    return pred[0]

# بناء واجهة المستخدم في Streamlit
st.title("🫁 Lung Cancer Classification")
st.write("قم برفع صورة الأشعة لتشخيص الحالة واقتراح العلاج المناسب.")

# مكون رفع الملفات
uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # فتح الصورة وعرضها للمستخدم
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("تشخيص الصورة"):
        with st.spinner("جاري تحليل الصورة..."):
            pred_idx = model_predict(img, model)
            
            labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
            medicines = {
                'adenocarcinoma': 'Gefitinib or Erlotinib or Osimertinib',
                'large.cell.carcinoma': 'Platinum-based chemotherapy',
                'normal': 'No treatment needed',
                'squamous.cell.carcinoma': 'Cisplatin, Gemcitabine'
            }
            
            label = labels[pred_idx]
            medicine = medicines[label]
            
            # عرض النتائج في واجهة مستخدم واضحة
            st.divider()
            if label == 'normal':
                st.success(f"**النتيجة:** {label}")
            else:
                st.warning(f"**النتيجة:** {label}")
                
            st.info(f"**العلاج المقترح:** {medicine}")
