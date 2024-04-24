import React, { useState } from "react";
import axios from "axios";

function App() {
  const [inputData, setInputData] = useState("");
  const [predictions, setPredictions] = useState([]);

  const handleInputChange = (event) => {
    setInputData(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      console.log("SUBMIT");
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        {
          input_data: inputData,
        }
      );
      setPredictions(response.data);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <h1>Predict Tags</h1>
      <textarea value={inputData} onChange={handleInputChange} />
      <br />
      <button onClick={handleSubmit}>Predict</button>
      <br />
      <h2>Predictions:</h2>
      <ul>
        {/* {predictions} */}
        {predictions.map((prediction, index) => (
          <li key={index}>{prediction}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;
