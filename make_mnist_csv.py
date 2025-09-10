import os, gzip, struct, urllib.request
import numpy as np

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_if_needed(filename, d="data"):
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    if not os.path.exists(path):
        print("Downloading", filename)
        urllib.request.urlretrieve(BASE + filename, path)
    return path

def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Bad magic for images"
        data = np.frombuffer(f.read(rows*cols*num), dtype=np.uint8)
        return data.reshape(num, rows*cols)

def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Bad magic for labels"
        return np.frombuffer(f.read(num), dtype=np.uint8)

def write_csv(images, labels, out_path):
    assert images.shape[0] == labels.shape[0]
    with open(out_path, "w") as f:
        for i in range(images.shape[0]):
            row = ",".join([str(int(labels[i]))] + [str(int(v)) for v in images[i]])
            f.write(row + "\n")
    print("Wrote", out_path)

def main():
    ti = read_idx_images(download_if_needed(FILES["train_images"]))
    tl = read_idx_labels(download_if_needed(FILES["train_labels"]))
    te = read_idx_images(download_if_needed(FILES["test_images"]))
    tel = read_idx_labels(download_if_needed(FILES["test_labels"]))
    write_csv(ti, tl, "mnist_train.csv")
    write_csv(te, tel, "mnist_test.csv")

if __name__ == "__main__":
    main()