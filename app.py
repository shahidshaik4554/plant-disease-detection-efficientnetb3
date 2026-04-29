import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

MODEL_PATH = r"C:\Users\Dell\Desktop\final\model\effb3_13classes_final.keras"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

CLASS_NAMES = list(model.class_names) if hasattr(model, "class_names") else [
    "Apple_healthy","Apple_rust","Apple_scab",
    "Corn_gray_leaf_spot","Corn_healthy","Corn_leaf_blight","Corn_rust",
    "Tomato_bacterial","Tomato_early_blight","Tomato_healthy",
    "Tomato_late_blight","Tomato_mosaic_virus","Tomato_yellow_virus"
]

IMG_SIZE = 224
LAST_CONV_LAYER = "top_conv"   # EfficientNetB3 last conv layer

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def make_gradcam_heatmap(img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    heatmap_path = os.path.join(UPLOAD_FOLDER, "gradcam.jpg")
    cv2.imwrite(heatmap_path, superimposed)

    return heatmap_path

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img_array = preprocess_img(filepath)
    preds = model.predict(img_array)[0]

    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(CLASS_NAMES[i], float(preds[i])) for i in top3_idx]

    heatmap = make_gradcam_heatmap(img_array)
    heatmap_path = overlay_heatmap(filepath, heatmap)

    return jsonify({
        "prediction": top3[0][0],
        "confidence": float(top3[0][1]),
        "top3": top3,
        "image": filepath,
        "gradcam": heatmap_path
    })

if __name__ == "__main__":
    app.run(debug=False, port=5001)