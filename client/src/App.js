import React, { useState } from "react";
import axios from "axios";
import Navbar from "./components/NavBar";

function App() {
  const [question, setQuestion] = useState("");
  const [description, setDescription] = useState("");
  const [previousAction, setPreviousAction] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [similar, setSimilar] = useState([]);

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleDescriptionChange = (event) => {
    setDescription(event.target.value);
  };

  const handlePreviousActionChange = (event) => {
    setPreviousAction(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      console.log("SUBMIT");
      const inputDataToSend = `${question} ${description} ${previousAction}`; // Combine all three inputs
      const predictResponse = await axios.post(
        "http://127.0.0.1:5000/predict",
        {
          input_data: inputDataToSend,
        }
      );
      console.log(predictResponse.data.prediction[0]);
      setPredictions(predictResponse.data.prediction[0]);
  
      const similarResponse = await axios.post(
        "http://127.0.0.1:5000/similar_questions",
        {
          input_data: inputDataToSend,
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
      <label>Question:</label>
      <br />
      <input type="text" value={question} onChange={handleQuestionChange} />
      <br />
      <label>Description:</label>
      <br />
      <input type="text" value={description} onChange={handleDescriptionChange} />
      <br />
      <label>Previous Action:</label>
      <br />
      <input type="text" value={previousAction} onChange={handlePreviousActionChange} />
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
