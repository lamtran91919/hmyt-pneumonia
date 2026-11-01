"""
Build a small, balanced NIH ChestX-ray14 pneumonia subset WITHOUT downloading the 45 GB Kaggle copy.

Source: Hugging Face "chehablab/NIHChestXR" (CC0, ~2.5 GB parquet, 224px images, patient ids, labels).
The dataset is STREAMED: nothing is cached except the images we keep (a few tens of MB).

    pip install datasets pillow
    python prepare_nih_subset.py --per-class 1000 --out nih_subset

Output:  nih_subset/images/*.jpg   and   nih_subset/metadata.csv  (path,label,patient_id,scan_id,age,sex)
label 1 = Pneumonia appears in the label list, label 0 = "No Finding" only.
Both classes are sampled uniformly over the whole dataset (reservoir sampling), not just the first patients.
NIH labels are NLP-mined from reports, so treat them as noisy.
Attribution (required by NIH): https://nihcc.app.box.com/v/ChestXray-NIHCC, data provided by the NIH Clinical Center.
"""
import argparse, csv, random
from pathlib import Path

PNEUMONIA, NO_FINDING = 7, 0   # label indices from the dataset card


def select(rows, per_class, seed=42):
    """rows: iterable of dicts with labels, image{'bytes'}, patient_id, scan_id, age, sex.
    Returns (normal_rows, pneumonia_rows), each at most per_class long, uniformly sampled."""
    rng = random.Random(seed)
    pneu, normal, seen_normal, seen = [], [], 0, 0
    for r in rows:
        seen += 1
        if seen % 10000 == 0:
            print(f"  scanned {seen} rows | pneumonia seen: {len(pneu)} | no-finding seen: {seen_normal}", flush=True)
        labels = list(r["labels"])
        slim = {k: r[k] for k in ("image", "patient_id", "scan_id", "age", "sex")}
        if PNEUMONIA in labels:
            pneu.append(slim)
        elif labels == [NO_FINDING]:
            seen_normal += 1
            if len(normal) < per_class:
                normal.append(slim)
            else:                                   # reservoir sampling: uniform over the whole stream
                j = rng.randrange(seen_normal)
                if j < per_class:
                    normal[j] = slim
    if len(pneu) > per_class:
        pneu = rng.sample(pneu, per_class)
    return normal, pneu


def save(normal, pneu, out):
    out = Path(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "patient_id", "scan_id", "age", "sex"])
        for y, group in ((0, normal), (1, pneu)):
            for r in group:
                b = r["image"]["bytes"]
                ext = ".jpg" if b[:2] == b"\xff\xd8" else ".png"
                rel = f"images/{r['patient_id']}_{r['scan_id']}_{y}{ext}"
                (out / rel).write_bytes(b)
                w.writerow([rel, y, r["patient_id"], r["scan_id"], r["age"], r["sex"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--out", default="nih_subset")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from datasets import load_dataset, Image
    ds = load_dataset("chehablab/NIHChestXR", split="train", streaming=True)
    ds = ds.cast_column("image", Image(decode=False))     # keep raw bytes; no decoding of the 100k images we discard
    normal, pneu = select(ds, a.per_class, a.seed)
    save(normal, pneu, a.out)
    print(f"saved {len(normal)} No-Finding + {len(pneu)} Pneumonia images to {a.out}/")
    if len(pneu) < a.per_class:
        print(f"note: only {len(pneu)} pneumonia images exist; consider --per-class {len(pneu)} for balance")


if __name__ == "__main__":
    main()
