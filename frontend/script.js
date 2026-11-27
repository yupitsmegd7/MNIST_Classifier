"use strict";

const form = document.querySelector("#predict-form");
const dropZone = document.querySelector("#drop-zone");
const fileInput = document.querySelector("#image-file");
const browseButton = document.querySelector("#browse-button");
const sampleButton = document.querySelector("#sample-button");
const predictButton = document.querySelector("#predict-button");
const predictLabel = document.querySelector("#predict-button-label");
const uploadPrompt = document.querySelector("#upload-prompt");
const previewPanel = document.querySelector("#preview-panel");
const preview = document.querySelector("#preview");
const fileName = document.querySelector("#file-name");
const inputError = document.querySelector("#input-error");
const resultWell = document.querySelector("#result-well");
const resultValue = document.querySelector("#result-value");
const resultStatus = document.querySelector("#result-status");

const imageTypes = new Set(["image/png", "image/jpeg", "image/webp"]);
const maxBytes = 12 * 1024 * 1024;
let preparedImage = null;
let previewUrl = null;
let busy = false;

function setResult(state, value, message) {
  resultWell.dataset.state = state;
  resultValue.textContent = value;
  resultStatus.textContent = message;
}

function showInputError(message) {
  inputError.textContent = message;
  inputError.hidden = !message;
}

function setBusy(value, label = "Predict digit") {
  busy = value;
  browseButton.disabled = value;
  sampleButton.disabled = value;
  predictButton.disabled = value || !preparedImage;
  predictLabel.textContent = label;
}

function canvasFor(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function pngFromCanvas(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("The image could not be prepared."));
    }, "image/png");
  });
}

function grayscale(pixels, offset) {
  return .299 * pixels[offset] + .587 * pixels[offset + 1] + .114 * pixels[offset + 2];
}

function borderColors(pixels, width, height) {
  const lightness = [];
  const opacity = [];

  // Sample near the edges so that a centered digit has little effect on the estimate.
  for (let sample = 0; sample < 12; sample++) {
    const x = Math.round(sample * (width - 1) / 11);
    const y = Math.round(sample * (height - 1) / 11);
    const offsets = [x, (height - 1) * width + x, y * width, y * width + width - 1];
    for (const index of offsets) {
      const offset = index * 4;
      lightness.push(grayscale(pixels, offset));
      opacity.push(pixels[offset + 3]);
    }
  }

  lightness.sort((a, b) => a - b);
  opacity.sort((a, b) => a - b);
  return { light: lightness[24] > 127, transparent: opacity[24] < 128 };
}

async function prepareImage(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("This image could not be opened. Try another PNG, JPG, or WebP.");
  }

  const scale = Math.min(1, 640 / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const source = canvasFor(width, height);
  const context = source.getContext("2d", { willReadFrequently: true });
  try {
    context.drawImage(bitmap, 0, 0, width, height);
  } finally {
    bitmap.close();
  }

  const image = context.getImageData(0, 0, width, height);
  const pixels = image.data;
  const background = borderColors(pixels, width, height);
  let left = width;
  let top = height;
  let right = -1;
  let bottom = -1;
  let darkest = 255;
  let brightest = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 4;
      const alpha = pixels[offset + 3];
      const gray = grayscale(pixels, offset);
      let ink = background.light ? 255 - gray : gray;

      if (background.transparent) {
        // A single-color stroke on a transparent canvas is defined by its opacity.
        ink = alpha;
      } else {
        ink *= alpha / 255;
      }

      pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = ink;
      pixels[offset + 3] = 255;
      darkest = Math.min(darkest, ink);
      brightest = Math.max(brightest, ink);

      if (ink > 32) {
        left = Math.min(left, x);
        top = Math.min(top, y);
        right = Math.max(right, x);
        bottom = Math.max(bottom, y);
      }
    }
  }

  if (right < 0 || brightest - darkest < 32) {
    throw new Error("No visible digit found. Try a digit that contrasts with its background.");
  }

  context.putImageData(image, 0, 0);

  // Preserve an original MNIST PNG so its predictions can be compared with Kaggle.
  if (width === 28 && height === 28 && !background.light && !background.transparent) {
    return pngFromCanvas(source);
  }

  const output = canvasFor(28, 28);
  const drawing = output.getContext("2d");
  drawing.fillStyle = "#000";
  drawing.fillRect(0, 0, 28, 28);

  const cropWidth = right - left + 1;
  const cropHeight = bottom - top + 1;
  const fit = 20 / Math.max(cropWidth, cropHeight);
  const digitWidth = Math.max(1, Math.round(cropWidth * fit));
  const digitHeight = Math.max(1, Math.round(cropHeight * fit));
  drawing.imageSmoothingQuality = "high";
  drawing.drawImage(
    source, left, top, cropWidth, cropHeight,
    Math.floor((28 - digitWidth) / 2), Math.floor((28 - digitHeight) / 2),
    digitWidth, digitHeight
  );

  return pngFromCanvas(output);
}

function clearImage() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute("src");
  preparedImage = null;
  uploadPrompt.hidden = false;
  previewPanel.hidden = true;
  browseButton.textContent = "Browse files";
  setResult("idle", "?", "Awaiting an image");
}

async function selectImage(file) {
  if (busy || !file) return;

  clearImage();
  showInputError("");
  if (!imageTypes.has(file.type) && (file.type || !/\.(png|jpe?g|webp)$/i.test(file.name))) {
    showInputError("Choose a PNG, JPG, or WebP image.");
    setBusy(false);
    return;
  }
  if (file.size > maxBytes) {
    showInputError("This file is too large. Choose an image under 12 MB.");
    setBusy(false);
    return;
  }

  setBusy(true, "Preparing image…");
  try {
    preparedImage = await prepareImage(file);
    previewUrl = URL.createObjectURL(preparedImage);
    preview.src = previewUrl;
    fileName.textContent = file.name;
    uploadPrompt.hidden = true;
    previewPanel.hidden = false;
    browseButton.textContent = "Change image";
    setResult("ready", "?", "Ready to predict");
  } catch (error) {
    showInputError(error.message);
  } finally {
    setBusy(false);
  }
}

async function sampleImage() {
  const canvas = canvasFor(80, 80);
  const drawing = canvas.getContext("2d");
  drawing.fillStyle = "#f7f7f7";
  drawing.fillRect(0, 0, 80, 80);
  drawing.strokeStyle = "#111";
  drawing.lineWidth = 7;
  drawing.lineCap = "round";
  drawing.lineJoin = "round";
  drawing.beginPath();
  drawing.moveTo(21, 20);
  drawing.quadraticCurveTo(34, 17, 57, 19);
  drawing.quadraticCurveTo(45, 34, 32, 62);
  drawing.stroke();

  const sample = await pngFromCanvas(canvas);
  await selectImage(new File([sample], "Sample digit 7.png", { type: "image/png" }));
}

browseButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectImage(fileInput.files[0]);
  fileInput.value = "";
});
sampleButton.addEventListener("click", () => {
  sampleImage().catch((error) => showInputError(error.message));
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (busy) return;
  event.dataTransfer.dropEffect = "copy";
  dropZone.classList.add("is-dragging");
});
dropZone.addEventListener("dragleave", (event) => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("is-dragging");
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  selectImage(event.dataTransfer.files[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!preparedImage || busy) return;

  showInputError("");
  setResult("loading", "…", "Running the model");
  setBusy(true, "Predicting…");

  const body = new FormData();
  body.append("image", preparedImage, "digit.png");

  try {
    let response;
    try {
      response = await fetch("/predict", { method: "POST", body });
    } catch {
      throw new Error("Could not reach FastAPI. Check that the server is running.");
    }

    const result = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = typeof result?.detail === "string" ? result.detail : null;
      throw new Error(detail || `Prediction failed (HTTP ${response.status}).`);
    }

    if (!Number.isInteger(result?.label) || result.label < 0 || result.label > 9) {
      throw new Error("The server did not return a digit from 0 to 9.");
    }

    setResult("success", result.label, `Predicted digit: ${result.label}`);
  } catch (error) {
    setResult("error", "!", "Prediction unavailable");
    showInputError(error.message);
  } finally {
    setBusy(false);
  }
});
