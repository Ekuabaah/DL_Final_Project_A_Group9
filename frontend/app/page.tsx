"use client";

import Image from "next/image";
import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Prediction = {
  label: string;
  confidence: number;
  isUnknown: boolean;
  threshold: number;
  topProbabilities: { label: string; value: number }[];
};

type ApiPrediction = {
  label: string;
  confidence: number;
  isUnknown: boolean;
  threshold: number;
  topProbabilities: { label: string; value: number }[];
};

function prettyClassLabel(label: string) {
  return label
    .replace(/___/g, " - ")
    .replace(/_/g, " ")
    .replace(/\(maize\)/g, "Maize")
    .replace(/\s+/g, " ")
    .trim();
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const hasFile = useMemo(() => Boolean(selectedFile), [selectedFile]);
  const thresholdPercent = ((prediction?.threshold ?? 0.5) * 100).toFixed(0);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFilePicked = (file: File | null) => {
    setPrediction(null);
    setApiError(null);
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
    setApiError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = (await response.json()) as
        | ApiPrediction
        | { detail?: string };

      if (!response.ok) {
        const detail =
          "detail" in payload && typeof payload.detail === "string"
            ? payload.detail
            : `Request failed (${response.status})`;
        throw new Error(detail);
      }

      const result = payload as ApiPrediction;
      setPrediction({
        label: result.label,
        confidence: result.confidence,
        isUnknown: result.isUnknown,
        threshold: result.threshold,
        topProbabilities: result.topProbabilities,
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Inference failed";
      setApiError(message);
      setPrediction(null);
    } finally {
      setIsClassifying(false);
    }
  };

  const clearSelection = () => {
    handleFilePicked(null);
    setPrediction(null);
    setApiError(null);
  };

  return (
    <div className="page-shell">
      <main className="app-card">
        <section className="headline-block appear">
          <div className="brand-row">
            <span className="brand-dot" aria-hidden="true" />
            <p className="eyebrow">PlantVillage Inference</p>
          </div>
          <h1>Leaf Disease Classifier</h1>
          <p className="subtitle">
            Upload. Classify. Read confidence instantly.
          </p>
          <div className="meta-chips">
            <span>Threshold {thresholdPercent}%</span>
            <span>FastAPI Connected</span>
            <span>Top-3 Scores</span>
          </div>
        </section>

        <section className="dashboard-grid">
          <div className="panel panel-upload appear-delayed">
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
                  <span className="upload-glyph" aria-hidden="true">
                    +
                  </span>
                  <h2>Drop leaf image</h2>
                  <p>or tap to browse</p>
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
                className="primary-btn cursor-pointer"
              >
                {isClassifying ? "Running inference..." : "Classify image"}
              </button>
              <button
                type="button"
                onClick={clearSelection}
                disabled={!hasFile || isClassifying}
                className="ghost-btn cursor-pointer"
              >
                Reset
              </button>
            </div>
          </div>

          <aside
            className="panel panel-results appear-stagger"
            aria-live="polite"
          >
            <div className="result-head">
              <p className="result-title">Results</p>
              <p className="threshold-pill">Threshold {thresholdPercent}%</p>
            </div>

            {!prediction && apiError ? (
              <p className="result-error">{apiError}</p>
            ) : !prediction ? (
              <p className="result-placeholder">
                Select an image to view prediction.
              </p>
            ) : (
              <>
                <div
                  className={`status-banner ${prediction.isUnknown ? "status-unknown" : "status-known"}`}
                >
                  <p className="status-label">
                    {prediction.isUnknown ? "Unknown class" : "Best match"}
                  </p>
                  <h3>
                    {prediction.isUnknown
                      ? "Out of known class space"
                      : prettyClassLabel(prediction.label)}
                  </h3>
                  <p>Confidence {(prediction.confidence * 100).toFixed(1)}%</p>
                </div>

                <ul className="top-list">
                  {prediction.topProbabilities.map((item, index) => (
                    <li
                      key={item.label}
                      style={{ animationDelay: `${index * 90}ms` }}
                    >
                      <div className="prob-row-head">
                        <span>{prettyClassLabel(item.label)}</span>
                        <span>{(item.value * 100).toFixed(1)}%</span>
                      </div>
                      <div className="prob-track" aria-hidden="true">
                        <span
                          className="prob-fill"
                          style={{ width: `${Math.max(item.value * 100, 2)}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            )}

            <div className="mini-note">
              <p>Rule</p>
              <p>Below {thresholdPercent}% means outside known classes.</p>
            </div>
          </aside>
        </section>
      </main>
    </div>
  );
}
