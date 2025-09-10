import os, io, base64, numpy as np, gradio as gr
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=120)

# ----- CONFIG -----
INPUT = 784
H1, H2, OUT = 300, 100, 10
ART = "./artifacts"
TEST_CSV = "./mnist_test.csv"

# ----- LOADING -----
def load_bin(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != np.prod(shape):
        raise ValueError(f"{os.path.basename(path)} has {arr.size} floats, expected {np.prod(shape)} for shape {shape}")
    return arr.reshape(shape)

def load_params(art_dir=ART):
    W1 = load_bin(os.path.join(art_dir, "weights1.bin"), (INPUT, H1))
    b1 = load_bin(os.path.join(art_dir, "biases1.bin"),  (H1,))
    W2 = load_bin(os.path.join(art_dir, "weights2.bin"), (H1, H2))
    b2 = load_bin(os.path.join(art_dir, "biases2.bin"),  (H2,))
    W3 = load_bin(os.path.join(art_dir, "weights3.bin"), (H2, OUT))
    b3 = load_bin(os.path.join(art_dir, "biases3.bin"),  (OUT,))
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

def load_mnist_csv(path=TEST_CSV):
    labels, data = [], []
    with open(path, "r") as f:
        for line in f:
            toks = line.strip().split(",")
            if not toks or len(toks) < 785:
                continue
            labels.append(int(toks[0]))
            pix = np.array([float(x) for x in toks[1:785]], dtype=np.float32) / 255.0
            data.append(pix)
    X = np.stack(data, axis=0)   # [N, 784]
    y = np.array(labels, dtype=np.int64)
    return X, y

# ----- MODEL -----
def relu(x): return np.maximum(x, 0.0)
def softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def forward_np(X, P):
    z1 = X @ P["W1"] + P["b1"]; a1 = relu(z1)
    z2 = a1 @ P["W2"] + P["b2"]; a2 = relu(z2)
    z3 = a2 @ P["W3"] + P["b3"]
    return softmax(z3)

# ----- METRICS & VIS -----
def compute_accuracy(P, X, y, batch=2048):
    N = X.shape[0]
    pred = []
    for i in range(0, N, batch):
        prob = forward_np(X[i:i+batch], P)
        pred.append(prob.argmax(axis=1))
    pred = np.concatenate(pred)
    acc = (pred == y).mean()
    return float(acc), pred

def confusion_matrix(y_true, y_pred, K=10):
    cm = np.zeros((K, K), dtype=np.int32)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    return cm

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(10));   ax.set_yticks(range(10))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3]
    plt.close(fig)
    return img  # uint8 (H, W, 3)

def draw_to_28x28(np_img):
    # Convert RGB uint8 canvas to (28x28) float32 in [0,1], white digit on black
    img = Image.fromarray(np_img).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1), img

def blank_canvas(size=280):
    return (255 * np.ones((size, size, 3), dtype=np.uint8))

# Robustly convert whatever ImageEditor gives us into an RGB numpy array
def editor_value_to_np(value):
    if value is None:
        return None
    # Case 1: dict from ImageEditor
    if isinstance(value, dict):
        # Try 'image' first, then 'composite'
        candidate = value.get("image", None)
        if candidate is None:
            candidate = value.get("composite", None)
        # Sometimes it's a data URL string:
        if isinstance(candidate, str) and candidate.startswith("data:"):
            header, b64 = candidate.split(",", 1)
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return np.array(img)
        # Or already a PIL / numpy
        if isinstance(candidate, Image.Image):
            return np.array(candidate.convert("RGB"))
        if isinstance(candidate, np.ndarray):
            if candidate.ndim == 2:  # grayscale
                candidate = np.stack([candidate]*3, axis=-1)
            return candidate
        # Fallback: try background
        bg = value.get("background", None)
        if isinstance(bg, Image.Image):
            return np.array(bg.convert("RGB"))
        if isinstance(bg, np.ndarray):
            return bg
        return None
    # Case 2: PIL or numpy directly
    if isinstance(value, Image.Image):
        return np.array(value.convert("RGB"))
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            value = np.stack([value]*3, axis=-1)
        return value
    # Case 3: data URL
    if isinstance(value, str) and value.startswith("data:"):
        header, b64 = value.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return np.array(img)
    return None

# ----- STATE -----
STATE = {"P": None, "X": None, "y": None}

# ----- HANDLERS -----
def ui_load(art_dir, test_csv):
    try:
        STATE["P"] = load_params(art_dir)
        STATE["X"], STATE["y"] = load_mnist_csv(test_csv)
        return f"Loaded params from {art_dir}. Test set: {STATE['X'].shape[0]} samples."
    except Exception as e:
        STATE["P"] = STATE["X"] = STATE["y"] = None
        return f"Load error: {e}"

def ui_eval():
    P, X, y = STATE["P"], STATE["X"], STATE["y"]
    if P is None or X is None:
        return "Please load params and test set.", None
    acc, pred = compute_accuracy(P, X, y)
    cm = confusion_matrix(y, pred)
    img = plot_confusion(cm)
    return f"Test accuracy: {acc*100:.2f}%", img

def ui_predict_index(idx):
    P, X, y = STATE["P"], STATE["X"], STATE["y"]
    if P is None or X is None:
        return None, {"0": 0.0}, "Please load first."
    idx = int(np.clip(idx, 0, X.shape[0]-1))
    prob = forward_np(X[idx:idx+1], P)[0]
    img = (X[idx].reshape(28, 28) * 255).astype(np.uint8)
    img = Image.fromarray(img).convert("L").resize((224, 224), Image.NEAREST).convert("RGB")
    conf = {str(i): float(prob[i]) for i in range(10)}
    label = f"Pred: {int(np.argmax(prob))} | True: {int(y[idx])}"
    return img, conf, label

def ui_draw_predict(editor_value):
    # Always return (image, probs_dict, label_str)
    try:
        if STATE["P"] is None:
            return None, {}, "Please load first."
        arr = editor_value_to_np(editor_value)
        if arr is None:
            return None, {}, "Draw a digit, then click Predict."
        X, vis = draw_to_28x28(arr)
        prob = forward_np(X, STATE["P"])[0]
        conf = {str(i): float(prob[i]) for i in range(10)}
        label = f"Pred: {int(np.argmax(prob))}"
        return vis.resize((224, 224), Image.NEAREST).convert("RGB"), conf, label
    except Exception as e:
        return None, {}, f"Error: {e}"

# ----- UI -----
with gr.Blocks() as demo:
    gr.Markdown("# MNIST MLP — CUDA-trained model demo")

    with gr.Row():
        art_dir = gr.Textbox(value=ART, label="Artifacts dir (weights/biases .bin)")
        test_csv = gr.Textbox(value=TEST_CSV, label="mnist_test.csv path")
        load_btn = gr.Button("Load")
        status = gr.Markdown("")
    load_btn.click(ui_load, inputs=[art_dir, test_csv], outputs=status)

    with gr.Row():
        eval_btn = gr.Button("Evaluate on full test set")
        acc_md = gr.Markdown()
    cm_img = gr.Image(type="numpy", label="Confusion Matrix")
    eval_btn.click(ui_eval, outputs=[acc_md, cm_img])

    gr.Markdown("### Browse a test sample")
    with gr.Row():
        idx = gr.Slider(0, 9999, value=0, step=1, label="Test index")
        img_view = gr.Image(type="pil", label="Image")
        probs = gr.Label(num_top_classes=10, label="Probabilities")
        lbl = gr.Markdown()
    idx.change(ui_predict_index, inputs=idx, outputs=[img_view, probs, lbl])

    gr.Markdown("### Or draw a digit (then click Predict)")
    with gr.Row():
        editor = gr.ImageEditor(
            label="Draw here",
            value=blank_canvas(280),
            interactive=True,
            show_download_button=False,
            show_share_button=False,
            sources=[],  # no uploads: pure canvas
        )
        draw_img  = gr.Image(type="pil", label="Your digit")
        draw_probs = gr.Label(num_top_classes=10, label="Probabilities")
        draw_lbl   = gr.Markdown()
    predict_btn = gr.Button("Predict from drawing")
    predict_btn.click(ui_draw_predict, inputs=[editor], outputs=[draw_img, draw_probs, draw_lbl])

    # Best-effort live wiring (ignored if not supported)
    for event in ("edit", "change"):
        try:
            getattr(editor, event)(ui_draw_predict, inputs=[editor], outputs=[draw_img, draw_probs, draw_lbl])
        except Exception:
            pass

if __name__ == "__main__":
    demo.launch()