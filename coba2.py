import cv2
import numpy as np
import glob, os
from tqdm import tqdm
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------- util ----------
def read_image(path):
    img = cv2.imread(path)
    if img is None: raise ValueError(f"Failed load: {path}")
    return img

def segment_leaf(img):
    # resize opsional (cepat)
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    blur = cv2.GaussianBlur(img, (5,5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # threshold hijau (kisaran umum; nanti bisa tuning)
    lower = np.array([25, 40,  20], dtype=np.uint8)   # H,S,V
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # rapikan
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ambil kontur terbesar (daun utama)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, np.zeros(mask.shape, np.uint8)
    c = max(cnts, key=cv2.contourArea)
    leaf_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(leaf_mask, [c], -1, 255, -1)

    return img, leaf_mask

def percent_yellow_brown(img_bgr, mask):
    # Lab untuk intensitas kekuningan/cokelat
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    # piksel valid hanya area daun
    m = mask > 0
    if m.sum() == 0:
        return 0.0, 0.0
    # heuristik: kuning = b tinggi & a rendah-tinggi moderat; cokelat = L rendah & b sedang
    yellow = ((b > 150) & m).sum() / m.sum()
    brown  = ((L < 120) & (b > 120) & m).sum() / m.sum()
    return float(yellow), float(brown)

def color_stats(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    def masked_stats(ch):
        vals = ch[mask>0].astype(np.float32)
        if len(vals) == 0: return (0,0)
        return float(vals.mean()), float(vals.std())

    Hm,Hs = masked_stats(hsv[:,:,0]); Sm,Ss = masked_stats(hsv[:,:,1]); Vm,Vs = masked_stats(hsv[:,:,2])
    Lm,Ls = masked_stats(lab[:,:,0]); am,as_ = masked_stats(lab[:,:,1]); bm,bs = masked_stats(lab[:,:,2])
    y,b = percent_yellow_brown(img_bgr, mask)
    return [Hm,Hs,Sm,Ss,Vm,Vs,Lm,Ls,am,as_,bm,bs,y,b]

def texture_feats(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Normalisasi histogram sederhana untuk GLCM stabil
    g = gray.copy()
    if (mask>0).sum() == 0:
        return [0,0,0,0,0,0] + [0]*16

    g_masked = g[mask>0]
    # stretch to 8-bit 0..255
    if g_masked.std() > 1e-3:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = g.astype(np.uint8)

    # GLCM
    glcm = graycomatrix(g, distances=[2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
    glcm_feats = []
    for p in props:
        glcm_feats.append(float(graycoprops(glcm, p).mean()))

    # LBP
    P, R = 8, 1
    lbp = local_binary_pattern(g, P, R, method='uniform')
    lbp = lbp[mask>0]
    hist, _ = np.histogram(lbp, bins=np.arange(0, P+3), range=(0, P+2), density=True)
    return glcm_feats + hist.tolist()  # 6 + 10 = 16 features

def shape_feats(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return [0,0,0,0]
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 1
    solidity = (area / hull_area) if hull_area > 0 else 0
    x,y,w,h = cv2.boundingRect(c)
    extent = (area / (w*h)) if w*h>0 else 0
    return [float(area), float(peri), float(solidity), float(extent)]

def extract_features(path):
    img = read_image(path)
    img, mask = segment_leaf(img)
    feats = []
    feats += color_stats(img, mask)
    feats += texture_feats(img, mask)
    feats += shape_feats(mask)
    return np.array(feats, dtype=np.float32)

# ---------- load dataset from folders ----------
def load_dataset(root):
    X, y = [], []
    classes = [('sehat', 0), ('sakit', 1)]
    for cls_name, cls_id in classes:
        for p in glob.glob(os.path.join(root, cls_name, "*")):
            ext = os.path.splitext(p)[1].lower()
            if ext not in [".jpg",".jpeg",".png",".bmp",".webp"]: continue
            try:
                X.append(extract_features(p))
                y.append(cls_id)
            except Exception as e:
                print("skip:", p, e)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    train_root = "data/train"
    valid_root = "data/valid"

    Xtr, ytr = load_dataset(train_root)
    Xva, yva = load_dataset(valid_root)

    print("Train:", Xtr.shape, "Valid:", Xva.shape)

    # Dua alternatif model: SVM RBF dan RandomForest → pilih terbaik
    candidates = {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced")),
        ]),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced_subsample", random_state=42)
    }

    best_name, best_model, best_score = None, None, -1
    for name, model in candidates.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="f1")
        print(f"{name} 5-fold F1: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_score:
            best_name, best_model, best_score = name, model, scores.mean()

    # Fit di full train, evaluasi valid
    best_model.fit(Xtr, ytr)
    ypred = best_model.predict(Xva)
    print("\nBest model:", best_name)
    print(confusion_matrix(yva, ypred))
    print(classification_report(yva, ypred, target_names=["sehat","sakit"]))

    # Simpan model
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{best_name}_pakcoy.pkl")
    print("Saved:", f"models/{best_name}_pakcoy.pkl")
