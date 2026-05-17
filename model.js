
// =========================
// LABELS
// =========================

const DEFAULT_LABELS = [

  {
    id: "akiec",
    name: "Actinic keratosis",
    note: "Potentially precancerous lesion.",
    urgent: true
  },

  {
    id: "bcc",
    name: "Basal cell carcinoma",
    note: "Common skin cancer category.",
    urgent: true
  },

  {
    id: "bkl",
    name: "Benign keratosis-like lesion",
    note: "Usually benign lesion type.",
    urgent: false
  },

  {
    id: "df",
    name: "Dermatofibroma",
    note: "Typically benign skin lesion.",
    urgent: false
  },

  {
    id: "mel",
    name: "Melanoma",
    note: "Potentially dangerous lesion category.",
    urgent: true
  },

  {
    id: "nv",
    name: "Melanocytic nevus",
    note: "Mole-like lesion category.",
    urgent: false
  },

  {
    id: "scc",
    name: "Squamous cell carcinoma",
    note: "Potentially dangerous skin cancer.",
    urgent: true
  },

  {
    id: "vasc",
    name: "Vascular lesion",
    note: "Blood vessel related lesion.",
    urgent: false
  }

];


// =========================
// GLOBALS
// =========================

let session = null;

let labels = DEFAULT_LABELS;


// =========================
// LOAD MODEL
// =========================

export async function loadClassifier() {

  if (!window.ort) {

    return {
      kind: "demo"
    };

  }

  try {

    session =
      await window.ort.InferenceSession.create(
        "models/skin-lesion.onnx",
        {
          executionProviders: ["wasm"]
        }
      );

    return {
      kind: "onnx"
    };

  } catch (error) {

    console.log(
      "ONNX model not found. Using demo mode."
    );

    return {
      kind: "demo"
    };

  }

}


// =========================
// MAIN PREDICTION
// =========================

export async function classifyImage(imageElement) {

  // =====================
  // REAL MODEL
  // =====================

  if (session) {

    const tensor =
      imageToTensor(imageElement, 224);

    const inputName =
      session.inputNames[0];

    const output =
      await session.run({
        [inputName]: tensor
      });

    const outputName =
      session.outputNames[0];

    const scores =
      softmax(
        Array.from(
          output[outputName].data
        )
      );

    return toPredictions(scores, "onnx");

  }

  // =====================
  // DEMO MODE
  // =====================

  return demoClassify(imageElement);

}


// =========================
// IMAGE -> TENSOR
// =========================

function imageToTensor(imageElement, size) {

  const canvas =
    document.createElement("canvas");

  canvas.width = size;
  canvas.height = size;

  const context =
    canvas.getContext(
      "2d",
      {
        willReadFrequently: true
      }
    );

  context.drawImage(
    imageElement,
    0,
    0,
    size,
    size
  );

  const { data } =
    context.getImageData(
      0,
      0,
      size,
      size
    );

  const floatData =
    new Float32Array(
      size * size * 3
    );

  const mean =
    [0.485, 0.456, 0.406];

  const std =
    [0.229, 0.224, 0.225];

  for (
    let pixel = 0;
    pixel < size * size;
    pixel++
  ) {

    const offset = pixel * 4;

    floatData[offset / 4 * 3] = data[offset] / 255;

    floatData[offset / 4 * 3 + 1] = data[offset + 1] / 255;

    floatData[offset / 4 * 3 + 2] = data[offset + 2] / 255;

  }

  return new window.ort.Tensor(

    "float32",

    floatData,

    [1, size, size, 3]

  );

}


// =========================
// DEMO CLASSIFIER
// =========================

function demoClassify(imageElement) {

  const features =
    sampleImageFeatures(imageElement);

  const scores = [

    0.35
      + features.redness * 0.2
      + features.edgeDensity * 0.25,

    0.42
      + features.darkness * 0.18
      + features.edgeDensity * 0.16,

    0.55
      + features.warmth * 0.2,

    0.38
      + features.symmetry * 0.14,

    0.28
      + features.darkness * 0.28
      + features.irregularity * 0.3,

    0.6
      + features.symmetry * 0.22
      - features.irregularity * 0.12,

    0.34
      + features.redness * 0.26

  ];

  const normalized =
    softmax(scores);

  return toPredictions(
    normalized,
    "demo"
  );

}


// =========================
// IMAGE FEATURES
// =========================

function sampleImageFeatures(imageElement) {

  const size = 96;

  const canvas =
    document.createElement("canvas");

  canvas.width = size;
  canvas.height = size;

  const context =
    canvas.getContext(
      "2d",
      {
        willReadFrequently: true
      }
    );

  context.drawImage(
    imageElement,
    0,
    0,
    size,
    size
  );

  const { data } =
    context.getImageData(
      0,
      0,
      size,
      size
    );

  let red = 0;
  let green = 0;
  let blue = 0;

  let darkPixels = 0;

  for (
    let i = 0;
    i < data.length;
    i += 4
  ) {

    red += data[i];
    green += data[i + 1];
    blue += data[i + 2];

    const luminance =
      0.2126 * data[i]
      + 0.7152 * data[i + 1]
      + 0.0722 * data[i + 2];

    if (luminance < 92) {
      darkPixels++;
    }

  }

  const pixels =
    size * size;

  const avgRed =
    red / pixels;

  const avgGreen =
    green / pixels;

  const avgBlue =
    blue / pixels;

  return {

    redness:
      clamp01(
        (avgRed - avgGreen) / 80
      ),

    warmth:
      clamp01(
        (
          avgRed
          + avgGreen
          - avgBlue * 1.7
        ) / 180
      ),

    darkness:
      darkPixels / pixels,

    edgeDensity:
      Math.random() * 0.7,

    irregularity:
      Math.random() * 0.6,

    symmetry:
      Math.random()

  };

}


// =========================
// SOFTMAX
// =========================

function softmax(values) {

  const max =
    Math.max(...values);

  const exps =
    values.map(
      (value) =>
        Math.exp(value - max)
    );

  const sum =
    exps.reduce(
      (a, b) => a + b,
      0
    );

  return exps.map(
    (value) => value / sum
  );

}


// =========================
// FORMAT PREDICTIONS
// =========================

function toPredictions(
  scores,
  source
) {

  return labels

    .map((label, index) => ({

      ...label,

      probability:
        scores[index] ?? 0,

      source

    }))

    .sort(
      (a, b) =>
        b.probability - a.probability
    );

}


// =========================
// CLAMP
// =========================

function clamp01(value) {

  return Math.min(
    1,
    Math.max(0, value)
  );

}
