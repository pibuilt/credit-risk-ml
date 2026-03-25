import React, { useState } from "react";
import axios from "axios";

function App() {
  const [loanAmnt, setLoanAmnt] = useState("");
  const [income, setIncome] = useState("");
  const [dti, setDti] = useState("");
  const [fico, setFico] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
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

      setResult(response.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
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

      <button onClick={handleSubmit}>Predict</button>

      {result && (
        <div>
          <h3>Result:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
