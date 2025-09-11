# Cuda-OCR (MNIST) — C++/CUDA training + Gradio demo

### Disclaimer
This reposiroty is a personal project I built to improve my C++ and CUDA skills.
It is not the same as my school submission; it reuses only the general idea and some baseline concepts from my OCR coursework.

### What this is

A from-scratch handwritten digit recognizer:

- **Training** implemented in **C++/CUDA**: forward & backward kernels for a fully-connected MLP (784→300→100→10), with ReLU (which was an issue with the baseline code I had given that in cuda the activation function should be non-linear), Softmax, Cross-Entropy, and SGD weight updates.

- **Visualization** in **Python/Gradio**: load exported weights, evaluate on MNIST test set, show accuracy + confusion matrix, browse samples, and draw a digit for live prediction.

### Highlights

- Custom CUDA kernels: forward, ReLU, softmax, cross_entropy, cross_entropy_backwards, backward, ReLU_backwards, update_layer.

- Deterministic weight init via cuRAND.

- Clean host–device error handling and simple timing utilities.

- Zero heavy dependencies for data: a tiny script downloads official MNIST IDX and writes CSVs.

--- 
### Getting started
### 1) Build and run the CUDA trainer
Requirements: NVCC + CUDA toolkit (+ NVIDIA driver)
```bash
# build
nvcc -O3 -std=c++17 main.cu -o main -lcurand

# generate MNIST CSVs (small Python script)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python make_mnist_csv.py  # writes mnist_train.csv / mnist_test.csv

# train (exports weights to ./artifacts/*.bin at the end)
./main
```
### 2) Launch the visual demo

```bash
# still in the venv
python viz_mnist_mlp.py
# open http://localhost:7860
# Click "Load", then "Evaluate". You can browse samples or draw a digit.
```

--- 
### How it works (high-level)

```rust
Host (C++)                   Device (CUDA)
-----------                  -----------------------------
init_rand          ->        cuRAND weight/bias init
forward            ->        Y = X @ W + b
ReLU               ->        A = max(0, Y)
softmax            ->        P = softmax(logits)
cross_entropy      ->        L = -∑ y*log(p)
cross_entropy_back ->        dL/dlogits = P - y
backward           ->        propagate gradients through layers
update_layer       ->        W -= lr * dW  ;  b -= lr * db

```

After training, the program saves raw float32 weights/biases into ./artifacts.
The Python app loads those binaries, runs NumPy inference, and provides an interactive UI.

---
### Troubleshooting and next commits
- NVCC link error → make sure you link cuRAND: -lcurand.
- Docker and better deploy to make testing for everyone easier
- Currently investigating with my professors if there might be a case of overfitting for the model. As such, it will be reviewed and renewed accordingly.