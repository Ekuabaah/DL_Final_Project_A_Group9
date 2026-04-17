"use client";

import Image from "next/image";
import { ChangeEvent, DragEvent, useMemo, useState } from "react";

const CLASS_LABELS = [
  "Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Corn_(maize)___Common_rust_",
  "Corn_(maize)___Northern_Leaf_Blight",
  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
  "Tomato___healthy",
] as const;

type Prediction = {
  label: string;
  confidence: number;
  isUnknown: boolean;
  topProbabilities: { label: string; value: number }[];
};

function seededRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function hashString(value: string) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function buildMockPrediction(file: File): Prediction {
  const seed = hashString(`${file.name}-${file.size}-${file.type}`);
  const rawScores = CLASS_LABELS.map(
    (_, index) => seededRandom(seed + index + 1) * 1.25 + 0.15,
  );

  const sum = rawScores.reduce((acc, current) => acc + current, 0);
  const probabilities = rawScores.map((value) => value / sum);

  const topIndex = probabilities.reduce(
    (bestIdx, probability, currentIdx, list) =>
      probability > list[bestIdx] ? currentIdx : bestIdx,
    0,
  );

  const topValue = probabilities[topIndex];
  const sortedTop = probabilities
    .map((value, index) => ({ label: CLASS_LABELS[index], value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  return {
    label: CLASS_LABELS[topIndex],
    confidence: topValue,
    isUnknown: topValue < 0.5,
    topProbabilities: sortedTop,
  };
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);

  const hasFile = useMemo(() => Boolean(selectedFile), [selectedFile]);

  const handleFilePicked = (file: File | null) => {
    setPrediction(null);
    setSelectedFile(file);

    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return objectUrl;
    });
  };

  const onFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    handleFilePicked(file);
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0] ?? null;
    handleFilePicked(file);
  };

  const classifyImage = async () => {
    if (!selectedFile) return;

    setIsClassifying(true);

    try {
      // Backend integration point:
      // 1) upload selectedFile to your API
      // 2) map API response to Prediction shape
      await new Promise((resolve) => setTimeout(resolve, 700));
      setPrediction(buildMockPrediction(selectedFile));
    } finally {
      setIsClassifying(false);
    }
  };

  const clearSelection = () => {
    handleFilePicked(null);
    setPrediction(null);
  };

  return (
    <div className="page-shell">
      <main className="app-card">
        <header className="hero">
          <p className="eyebrow">PlantVillage CNN Classifier</p>
          <h1>Disease Classification Interface</h1>
          <p className="subtitle">
            Upload a plant photo to preview the model decision. If top
            confidence is below <strong>50%</strong>, the UI marks it as out of
            the known classes.
          </p>
        </header>

        <section className="workspace">
          <label
            className={`dropzone ${isDragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              event.stopPropagation();
              setIsDragging(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              event.stopPropagation();
              setIsDragging(false);
            }}
            onDrop={onDrop}
          >
            <input
              type="file"
              accept="image/*"
              onChange={onFileInput}
              className="sr-only"
            />

            {!previewUrl ? (
              <div className="dropzone-empty">
                <h2>Drop image here</h2>
                <p>or click to choose a file from your device</p>
              </div>
            ) : (
              <figure className="preview-wrap">
                <Image
                  src={previewUrl}
                  alt="Selected plant preview"
                  width={1280}
                  height={720}
                  unoptimized
                />
                <figcaption>
                  <span>{selectedFile?.name}</span>
                  <span>
                    {selectedFile
                      ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                      : ""}
                  </span>
                </figcaption>
              </figure>
            )}
          </label>

          <div className="actions">
            <button
              type="button"
              onClick={classifyImage}
              disabled={!hasFile || isClassifying}
              className="primary-btn"
            >
              {isClassifying ? "Classifying..." : "Classify image"}
            </button>
            <button
              type="button"
              onClick={clearSelection}
              disabled={!hasFile || isClassifying}
              className="ghost-btn"
            >
              Clear
            </button>
          </div>

          <section className="result-card" aria-live="polite">
            {!prediction ? (
              <p className="result-placeholder">
                Upload an image and run classification to view the prediction
                output.
              </p>
            ) : prediction.isUnknown ? (
              <div className="unknown-state">
                <h3>Unknown / Out of class</h3>
                <p>
                  Highest confidence is{" "}
                  {(prediction.confidence * 100).toFixed(1)}% , which is under
                  50%.
                </p>
              </div>
            ) : (
              <div className="known-state">
                <h3>{prediction.label}</h3>
                <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
              </div>
            )}

            {prediction && (
              <ul className="top-list">
                {prediction.topProbabilities.map((item) => (
                  <li key={item.label}>
                    <span>{item.label}</span>
                    <span>{(item.value * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
