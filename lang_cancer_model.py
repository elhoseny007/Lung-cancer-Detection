import os
import gdown
import tensorflow as tf

MODEL_PATH = "model_vgg16_layer.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1oLp4MgPSVvNHMXfreF0jnSwY8JX011Qw"
    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url=url,
        output=MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)
