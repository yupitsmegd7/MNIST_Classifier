# Digit Garden — MNIST Classifier

A handwritten digit recognition web app built with **PyTorch**, **FastAPI**, and **vanilla JavaScript**. Upload an image containing a single digit (0–9), preview the processed model input, and get a prediction from a compact convolutional neural network.

## Features

- Drag-and-drop or browse uploads for PNG, JPG, and WebP images.
- Browser-side preprocessing: background detection, grayscale conversion, polarity adjustment, cropping, and centering.
- Preview of the exact 28 × 28 image submitted to the API.
- Built-in sample digit for a quick demonstration.
- CPU inference using the included model weights.
- FastAPI serves both the frontend and prediction endpoint; no frontend build step is required.

## How it works

```text
Upload → Browser preprocessing → 28 × 28 PNG → POST /predict
       → Grayscale tensor → PyTorch CNN → Predicted digit → UI
```

The browser estimates the background from image borders and converts the digit into light strokes on a dark background. It crops the detected foreground, preserves its aspect ratio, and fits it within a 20 × 20 region centered on a 28 × 28 canvas. Existing 28 × 28 images with dark, opaque backgrounds skip cropping and recentering.

The backend converts the uploaded image to grayscale, resizes it to 28 × 28, scales pixels to `[0, 1]`, and adds a batch dimension. Inference runs with `model.eval()` and `torch.inference_mode()`. The class with the highest output score becomes the predicted digit.

## Model architecture

| Layer | Output shape, excluding batch |
|---|---|
| Grayscale input | 1 × 28 × 28 |
| Conv2d: 1 → 8, 3 × 3, padding 1 + ReLU | 8 × 28 × 28 |
| MaxPool2d: 2 × 2 | 8 × 14 × 14 |
| Conv2d: 8 → 32, 3 × 3, padding 1 + ReLU | 32 × 14 × 14 |
| MaxPool2d: 2 × 2 | 32 × 7 × 7 |
| Flatten | 1,568 |
| Linear: 1,568 → 10 | 10 class scores |

**18,106 trainable parameters.** Weights are loaded from `digit_cnn_v3.pt` onto the CPU at application startup.

## Run locally

You need Python, Git, and a browser. Python 3.10 or newer is a suggested starting point; dependency versions are not pinned in this repository.

### 1. Clone the repository

```bash
git clone https://github.com/yupitsmegd7/MNIST_Classifier.git
cd MNIST_Classifier
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install torch torchvision fastapi uvicorn pillow python-multipart
```

### 3. Start the application

Run from the repository root:

```bash
python -m uvicorn main:app --reload
```

Open [Digit Garden](http://127.0.0.1:8000) in your browser. Upload a clear image containing one digit, or click **Try a sample**, then **Predict digit**.

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API

### `POST /predict`

Send a `multipart/form-data` request with an image file in the **`image`** field.

```bash
curl -X POST http://127.0.0.1:8000/predict -F "image=@digit.png"
```

On Windows PowerShell, use `curl.exe` if `curl` resolves to a PowerShell alias.

Example response:

```json
{"label": 7}
```

An unreadable image returns HTTP `400` with `{"detail":"Upload a valid image."}`. A missing image field returns HTTP `422`.

**Direct API callers should prepare a centered, light-on-dark digit image.** The API performs grayscale conversion and resizing, but cropping, background inversion, and centering happen in the browser. The browser's 12 MB upload limit is also a client-side check.

## Repository structure

```text
MNIST_Classifier/
├── main.py              # CNN definition, weight loading, API, static hosting
├── digit_cnn_v3.pt       # Saved model weights
├── frontend/
│   ├── index.html       # Upload and prediction interface
│   ├── script.js        # Image preprocessing and API requests
│   ├── styles.css      # Interface styling
│   └── frog.png         # Decorative illustration
├── LICENSE
└── README.md
```

## Scope and limitations

- Recognizes one digit per image; multi-digit recognition and handwriting transcription are outside the current scope.
- Returns a class label, without confidence scores or an unknown-input class. Non-digit images can still receive a digit prediction.
- Background detection uses heuristics, so cluttered photographs, shadows, and unusual handwriting may reduce reliability.
- The repository includes inference code and weights, but no training script, training configuration, or evaluation report. No test accuracy is claimed here.

## Possible improvements

- Add reproducible training and evaluation scripts with a confusion matrix and per-class metrics.
- Share preprocessing between browser and API clients.
- Add server-side upload limits and input validation.
- Pin dependencies and add API tests.

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Gourav Dutta.
