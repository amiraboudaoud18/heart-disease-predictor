import { useState } from "react";
import axios from "axios";
import "./App.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const fields = [
  { key: "age", label: "Age", hint: "years (e.g. 52)" },
  { key: "sex", label: "Sex", hint: "0 = Female, 1 = Male" },
  { key: "cp", label: "Chest Pain Type", hint: "0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic" },
  { key: "trestbps", label: "Resting Blood Pressure", hint: "mm Hg (e.g. 120)" },
  { key: "chol", label: "Serum Cholesterol", hint: "mg/dl (e.g. 200)" },
  { key: "fbs", label: "Fasting Blood Sugar > 120 mg/dl", hint: "0 = No, 1 = Yes" },
  { key: "restecg", label: "Resting ECG Results", hint: "0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy" },
  { key: "thalach", label: "Max Heart Rate Achieved", hint: "bpm (e.g. 150)" },
  { key: "exang", label: "Exercise Induced Angina", hint: "0 = No, 1 = Yes" },
  { key: "oldpeak", label: "ST Depression", hint: "e.g. 1.0" },
  { key: "slope", label: "Slope of Peak Exercise ST", hint: "0 = upsloping, 1 = flat, 2 = downsloping" },
  { key: "ca", label: "Major Vessels (Fluoroscopy)", hint: "0 – 4" },
  { key: "thal", label: "Thalassemia", hint: "0 = normal, 1 = fixed defect, 2 = reversible defect" },
];

const emptyForm = Object.fromEntries(fields.map((f) => [f.key, ""]));

export default function App() {
  const [form, setForm] = useState(emptyForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, parseFloat(v)])
      );
      const res = await axios.post(`${BACKEND_URL}/predict`, payload);
      setResult(res.data);
    } catch (err) {
      setError("Could not reach the prediction service. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm(emptyForm);
    setResult(null);
    setError(null);
  };

  return (
    <div className="page">
      <header className="top-bar">
        <div className="top-bar-inner">
          <div className="brand">
            <span className="brand-cross">+</span>
            <span className="brand-name">CardioScreen</span>
          </div>
          <span className="top-bar-subtitle">Heart Disease Risk Assessment</span>
        </div>
      </header>

      <main className="container">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Patient Clinical Parameters</h2>
            <p className="card-desc">
              Fill in the patient data below and click "Run Assessment" to receive a cardiovascular risk prediction.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="form">
            <div className="fields-grid">
              {fields.map((field) => (
                <div key={field.key} className="field">
                  <label className="label">{field.label}</label>
                  <input
                    className="input"
                    type="number"
                    step="any"
                    placeholder={field.hint}
                    value={form[field.key]}
                    onChange={(e) => handleChange(field.key, e.target.value)}
                    required
                  />
                  <span className="hint">{field.hint}</span>
                </div>
              ))}
            </div>

            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={handleReset}>
                Reset
              </button>
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? "Analyzing..." : "Run Assessment"}
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div className="alert alert-error">{error}</div>
        )}

        {result && (
          <div className={`result-card ${result.prediction === 1 ? "high" : "low"}`}>
            <div className="result-top">
              <div>
                <div className="result-label">Risk Classification</div>
                <div className="result-value">{result.label}</div>
              </div>
              <div className="result-prob-block">
                <div className="result-label">Disease Probability</div>
                <div className="result-prob">
                  {result.probability !== null
                    ? `${(result.probability * 100).toFixed(1)}%`
                    : "N/A"}
                </div>
              </div>
            </div>

            <div className="prob-track">
              <div
                className="prob-fill"
                style={{ width: `${((result.probability || 0) * 100).toFixed(1)}%` }}
              />
            </div>

            <p className="result-disclaimer">
              This result is generated by a machine learning model for reference purposes only.
              It does not constitute medical advice and should not replace professional clinical judgment.
            </p>
          </div>
        )}
      </main>

    
    </div>
  );
}