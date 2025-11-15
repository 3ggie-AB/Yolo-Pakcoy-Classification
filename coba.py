import sys, joblib, cv2
import numpy as np
from pakcoy_opencv_classic import segment_leaf, color_stats, texture_feats, shape_feats

model_path = "models/svm_rbf_pakcoy.pkl"  # atau rf_pakcoy.pkl
model = joblib.load(model_path)

def extract(img):
    img, mask = segment_leaf(img)
    feats = []
    feats += color_stats(img, mask)
    feats += texture_feats(img, mask)
    feats += shape_feats(mask)
    return np.array(feats, dtype=np.float32).reshape(1,-1)

img = cv2.imread(sys.argv[1])
X = extract(img)
pred = model.predict(X)[0]
prob = getattr(model, "predict_proba", lambda Z: [[0,0]])(X)[0]
label = ["sehat","sakit"][int(pred)]
print(label, prob)
