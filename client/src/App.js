import React, { useState } from "react";
import axios from "axios";
import Navbar from "./components/NavBar";

function App() {
  const [inputData, setInputData] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [similar, setSimilar] = useState([]);

  const handleInputChange = (event) => {
    setInputData(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      console.log("SUBMIT");
      const predictResponse = await axios.post(
        "http://127.0.0.1:5000/predict",
        {
          input_data: inputData,
        }
      );
      console.log(predictResponse.data.prediction[0]);
      setPredictions(predictResponse.data.prediction[0]);

      const similarResponse = await axios.post(
        "http://127.0.0.1:5000/similar_questions",
        {
          input_data: inputData,
        }
      );
      console.log(similarResponse.data.similar_questions);
      setSimilar(similarResponse.data.similar_questions);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <Navbar />
      <h1>Predict Tags</h1>
      <textarea value={inputData} onChange={handleInputChange} />
      <br />
      <button onClick={handleSubmit}>Predict</button>
      <br />
      <h2>Predictions:</h2>
      <ul>
        {predictions?.map((prediction, index) => (
          <li key={index}>{prediction}</li>
        ))}
      </ul>

      <h2>Similar Questions:</h2>
      <ol>
        {similar?.map((similarQuestion, index) => (
          <li key={index}>{similarQuestion}</li>
        ))}
      </ol>
    </div>
  );
}

export default App;
