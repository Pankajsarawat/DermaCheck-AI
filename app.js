
import { classifyImage, loadClassifier } from "./model.js";

const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");

const previewImage = document.querySelector("#previewImage");
const cameraVideo = document.querySelector("#cameraVideo");
const captureCanvas = document.querySelector("#captureCanvas");

const emptyState = document.querySelector("#emptyState");

const cameraButton = document.querySelector("#cameraButton");
const snapButton = document.querySelector("#snapButton");
const analyzeButton = document.querySelector("#analyzeButton");
const resetButton = document.querySelector("#resetButton");

const topLabel = document.querySelector("#topLabel");
const topDescription = document.querySelector("#topDescription");

const rankedList = document.querySelector("#rankedList");

const confidenceValue = document.querySelector("#confidenceValue");
const riskFill = document.querySelector("#riskFill");

const modelStatus = document.querySelector("#modelStatus");

let stream = null;
let currentImageUrl = null;

initialize();


// =========================
// INITIALIZE
// =========================

async function initialize() {

  try {

    await loadClassifier();

    modelStatus.textContent = "Model Ready";

  } catch (error) {

    modelStatus.textContent = "Demo Mode";

  }

}


// =========================
// FILE UPLOAD
// =========================

fileInput.addEventListener("change", (event) => {

  const [file] = event.target.files;

  if (file) {
    loadFile(file);
  }

});


// =========================
// DRAG DROP
// =========================

dropZone.addEventListener("dragover", (event) => {

  event.preventDefault();

  dropZone.classList.add("dragging");

});

dropZone.addEventListener("dragleave", () => {

  dropZone.classList.remove("dragging");

});

dropZone.addEventListener("drop", (event) => {

  event.preventDefault();

  dropZone.classList.remove("dragging");

  const [file] = event.dataTransfer.files;

  if (file && file.type.startsWith("image/")) {

    loadFile(file);

  }

});


// =========================
// CAMERA
// =========================

cameraButton.addEventListener("click", async () => {

  if (stream) {

    stopCamera();
    return;

  }

  try {

    stream = await navigator.mediaDevices.getUserMedia({

      video: {
        facingMode: "environment"
      },

      audio: false

    });

    cameraVideo.srcObject = stream;

    cameraVideo.hidden = false;

    previewImage.hidden = true;

    emptyState.hidden = true;

    cameraButton.textContent = "Stop Camera";

    snapButton.disabled = false;

  } catch (error) {

    topLabel.textContent = "Camera Error";

    topDescription.textContent =
      "Could not access camera.";

  }

});


// =========================
// CAPTURE IMAGE
// =========================

snapButton.addEventListener("click", () => {

  if (!stream) return;

  const width = cameraVideo.videoWidth;
  const height = cameraVideo.videoHeight;

  captureCanvas.width = width;
  captureCanvas.height = height;

  captureCanvas
    .getContext("2d")
    .drawImage(cameraVideo, 0, 0, width, height);

  captureCanvas.toBlob((blob) => {

    if (!blob) return;

    loadFile(
      new File(
        [blob],
        "capture.jpg",
        {
          type: "image/jpeg"
        }
      )
    );

    stopCamera();

  });

});


// =========================
// ANALYZE
// =========================

analyzeButton.addEventListener("click", async () => {

  if (previewImage.hidden) return;

  setBusy(true, "Analyzing...");

  try {

    const predictions =
      await classifyImage(previewImage);

    renderPredictions(predictions);

  } catch (error) {

    topLabel.textContent = "Analysis Failed";

    topDescription.textContent =
      error.message;

  } finally {

    setBusy(false);

  }

});


// =========================
// RESET
// =========================

resetButton.addEventListener("click", () => {

  stopCamera();

  clearImage();

  rankedList.innerHTML = "";

  topLabel.textContent =
    "No image analyzed";

  topDescription.textContent =
    "AI prediction results will appear here.";

  confidenceValue.textContent = "0%";

  riskFill.style.width = "0%";

});


// =========================
// LOAD FILE
// =========================

function loadFile(file) {

  stopCamera();

  if (currentImageUrl) {

    URL.revokeObjectURL(currentImageUrl);

  }

  currentImageUrl =
    URL.createObjectURL(file);

  previewImage.src = currentImageUrl;

  previewImage.hidden = false;

  cameraVideo.hidden = true;

  emptyState.hidden = true;

  analyzeButton.disabled = false;

  topLabel.textContent =
    "Ready to Analyze";

  topDescription.textContent =
    "Click analyze button to start prediction.";

}


// =========================
// RENDER RESULTS
// =========================

function renderPredictions(predictions) {

  const top = predictions[0];

  const percent =
    Math.round(top.probability * 100);

  topLabel.textContent = top.name;

  topDescription.textContent =
    top.note;

  confidenceValue.textContent =
    `${percent}%`;

  riskFill.style.width =
    `${percent}%`;

  rankedList.innerHTML =
    predictions.map((prediction) => {

      const rowPercent =
        Math.round(prediction.probability * 100);

      return `

      <div class="prediction-row ${prediction.urgent ? "warning" : ""}">

        <div class="prediction-head">

          <span>${prediction.name}</span>

          <span>${rowPercent}%</span>

        </div>

        <div class="bar">
          <span style="width:${rowPercent}%"></span>
        </div>

        <p class="prediction-note">
          ${prediction.note}
        </p>

      </div>

      `;

    }).join("");

}


// =========================
// CLEAR IMAGE
// =========================

function clearImage() {

  if (currentImageUrl) {

    URL.revokeObjectURL(currentImageUrl);

  }

  currentImageUrl = null;

  previewImage.removeAttribute("src");

  previewImage.hidden = true;

  cameraVideo.hidden = true;

  emptyState.hidden = false;

  analyzeButton.disabled = true;

  fileInput.value = "";

}


// =========================
// STOP CAMERA
// =========================

function stopCamera() {

  if (!stream) return;

  stream
    .getTracks()
    .forEach((track) => track.stop());

  stream = null;

  cameraVideo.srcObject = null;

  cameraVideo.hidden = true;

  cameraButton.textContent =
    "Start Camera";

  snapButton.disabled = true;

}


// =========================
// BUSY STATE
// =========================

function setBusy(isBusy, label = "Analyze") {

  analyzeButton.textContent = label;

  analyzeButton.disabled =
    isBusy || previewImage.hidden;

  fileInput.disabled = isBusy;

  cameraButton.disabled = isBusy;

  snapButton.disabled =
    isBusy || !stream;

}
