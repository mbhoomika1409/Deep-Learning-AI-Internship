# Waste Image Classification System (Major Project)

A small CNN that classifies a photo into one of 5 waste categories: **cardboard, glass, metal, paper, plastic**, with a simple Gradio web app.

**Important:** model scores do not guarantee correctness. Test accuracy is about 79.5%, and 6 of 352 test photos were wrong with a score of 90% or higher. See the limitations below.

## Results (test set, 352 images)

| Model | Test accuracy |
|---|---:|
| A: baseline CNN (2 plain conv blocks) | 65.1% |
| B: improved CNN (BatchNorm, global pooling, Dropout) | **79.5%** (final model) |
| C: improved CNN + data augmentation | 72.7% |

Full details, graphs, confusion matrix and error analysis: [`../Assignment/evaluation_report.md`](../Assignment/evaluation_report.md)

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads TrashNet and saves the 5 classes into `dataset/raw/` |
| `build_dataset.py` | Finds duplicate photos and makes a leak-free 70/15/15 split (`splits.csv`) |
| `common.py` | Shared preprocessing, dataset class and models (used by training and the app) |
| `train.py` | Trains the 3 configurations, plots curves, saves `waste_cnn.pth` |
| `evaluate.py` | Test accuracy, per-class results, confusion matrix, confidence analysis, failure examples |
| `predict.py` | Loads the saved model and predicts on new images |
| `app.py` | Gradio demo |
| `waste_cnn.pth` | Trained final model (about 1 MB, included in this repo) |
| `checkpoints/` | Best weights of each of the 3 configurations |
| `images/` | Graphs and example images |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the app (uses the included trained model)

```bash
python app.py
```

Open the link it prints (usually http://127.0.0.1:7860), upload a photo, and you get the top 3 predictions with model scores. If the top score is below 60%, the app says "Not sure".

## Retrain from scratch (optional)

The dataset is not stored in this repo. This downloads it:

```bash
python prepare_data.py      # downloads TrashNet (dataset-resized.zip), keeps 5 classes
python build_dataset.py     # duplicate check and leak-free split
python train.py             # about 10 to 15 minutes on a MacBook Air (add a number for fewer epochs)
python evaluate.py          # test results and figures
```

## Data

TrashNet, `garythung/trashnet` on Hugging Face, MIT licence (Yang and Thung, Stanford). 2,390 images in 5 classes, 2,357 after removing duplicates. The sixth class, "trash", was left out because it has only 137 images.

## Limitations

- A model score is not a guarantee of correctness.
- Trained on only 2,357 photos, mostly one object on a white background. Cluttered or real-life photos will do worse.
- Only 5 classes. Anything else (food, people, other materials) is still forced into one of them.
- Glass is often confused with metal, and plastic with glass.
- Not for real waste-sorting or safety decisions.

## Demonstration video

(Add the link here.)
