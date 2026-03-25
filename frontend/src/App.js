import React, { useState } from "react";
import axios from "axios";

function App() {
  const [loanAmnt, setLoanAmnt] = useState("");
  const [income, setIncome] = useState("");
  const [dti, setDti] = useState("");
  const [fico, setFico] = useState("");
  const [result, setResult] = useState(null);

  // Add missing state hooks for error and loading at the top level
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setError("");

    // Validation
    if (!loanAmnt) {
      setError("Loan Amount is required");
      return;
    }
    if (!income) {
      setError("Annual Income is required");
      return;
    }
    if (!dti) {
      setError("DTI is required");
      return;
    }
    if (!fico) {
      setError("FICO Score is required");
      return;
    }

    if (isNaN(Number(loanAmnt)) || Number(loanAmnt) <= 0) {
      setError("Loan Amount must be a positive number");
      return;
    }
    if (isNaN(Number(income)) || Number(income) <= 0) {
      setError("Annual Income must be a positive number");
      return;
    }
    if (isNaN(Number(dti)) || Number(dti) < 0) {
      setError("DTI must be a non-negative number");
      return;
    }
    if (isNaN(Number(fico)) || Number(fico) < 0) {
      setError("FICO Score must be a non-negative number");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post("/api/v1/predict", {
        data: [
          {
            loan_amnt: Number(loanAmnt),
            annual_inc: Number(income),
            dti: Number(dti),
            fico_range_low: Number(fico),
          },
        ],
      });

      const prediction = response.data.predictions[0];
      setResult(prediction);
    } catch (err) {
      console.error(err);
      setError("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Credit Risk Predictor</h1>

      <input
        placeholder="Loan Amount"
        value={loanAmnt}
        onChange={(e) => setLoanAmnt(e.target.value)}
      />
      <br />
      <br />
      {error && <p style={{ color: "red" }}>{error}</p>}

      <input
        placeholder="Annual Income"
        value={income}
        onChange={(e) => setIncome(e.target.value)}
      />
      <br />
      <br />

      <input
        placeholder="DTI"
        value={dti}
        onChange={(e) => setDti(e.target.value)}
      />
      <br />
      <br />

      <input
        placeholder="FICO Score"
        value={fico}
        onChange={(e) => setFico(e.target.value)}
      />
      <br />
      <br />

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && (
        <div>
          <h3>Result:</h3>
          {result && (
            <div>
              <h3>Result:</h3>
              <p>
                <strong>Risk Score:</strong> {result.risk_score}
              </p>
              <p>
                <strong>Risk Level:</strong> {result.risk_level}
              </p>
              <p>
                <strong>Default Probability:</strong>{" "}
                {result.default_probability}
              </p>
              <p>
                <strong>Risk Cluster:</strong> {result.risk_cluster}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
