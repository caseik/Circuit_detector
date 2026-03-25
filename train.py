import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import onnx
from onnx import helper, TensorProto, numpy_helper

IMG_SIZE = 64
DATASET_PATH = "elements/"
ONNX_MODEL_PATH = "model.onnx"
LABELS_PATH = "labels.txt"

hog = cv2.HOGDescriptor(
    (64, 64),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)

def get_label(filename):
    if filename.startswith("r"):
        return "R"
    elif filename.startswith("c"):
        return "C"
    elif filename.startswith("l"):
        return "L"
    return "unknown"

def preprocess(img):
    # Match onnx_classifier.cpp: 64x64 → Otsu INV → Canny (sin blur).
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return img

X, y = [], []
feature_dim = None

for f in os.listdir(DATASET_PATH):
    if not f.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(DATASET_PATH, f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue


    img = preprocess(img)
    features = hog.compute(img).reshape(-1).astype(np.float32)

    X.append(features)
    y.append(get_label(f))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=str)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual=True)
model.fit(X, y_encoded)

y_train_pred = model.predict(X)
train_acc = accuracy_score(y_encoded, y_train_pred)


# Export ONNX using only standard ops that OpenCV DNN supports.
# scores = MatMul(input, W) + b
# label  = ArgMax(scores, axis=1)
weights = model.coef_.astype(np.float32).T  # [n_features, n_classes]
bias = model.intercept_.astype(np.float32)  # [n_classes]

input_name = "input"
scores_name = "scores"
label_name = "label"

input_tensor = helper.make_tensor_value_info(
    input_name, TensorProto.FLOAT, [None, X.shape[1]]
)
scores_tensor = helper.make_tensor_value_info(
    scores_name, TensorProto.FLOAT, [None, len(label_encoder.classes_)]
)
label_tensor = helper.make_tensor_value_info(
    label_name, TensorProto.INT64, [None]
)

w_init = numpy_helper.from_array(weights, name="W")
b_init = numpy_helper.from_array(bias, name="B")

matmul_node = helper.make_node("MatMul", [input_name, "W"], ["mm"])
add_node = helper.make_node("Add", ["mm", "B"], [scores_name])
argmax_node = helper.make_node(
    "ArgMax",
    [scores_name],
    [label_name],
    axis=1,
    keepdims=0
)

graph = helper.make_graph(
    nodes=[matmul_node, add_node, argmax_node],
    name="linear_svm_as_dense",
    inputs=[input_tensor],
    outputs=[label_tensor, scores_tensor],
    initializer=[w_init, b_init]
)

onnx_model = helper.make_model(
    graph,
    producer_name="circuit_detector_train",
    opset_imports=[helper.make_operatorsetid("", 13)]
)
onnx.checker.check_model(onnx_model)

with open(ONNX_MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for label in label_encoder.classes_:
        f.write(f"{label}\n")

print(f"ONNX model saved to {ONNX_MODEL_PATH}")
print(f"Labels saved to {LABELS_PATH}")