import React, { useState } from "react";
import axios from "axios";
import Navbar from "./components/NavBar";

function App() {
  const [question, setQuestion] = useState("");
  const [description, setDescription] = useState("");
  const [previousAction, setPreviousAction] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [similar, setSimilar] = useState([]);
  const [selectedVectorizer, setSelectedVectorizer] = useState("tfidf+word2vec"); // Default value

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleDescriptionChange = (event) => {
    setDescription(event.target.value);
  };

  const handlePreviousActionChange = (event) => {
    setPreviousAction(event.target.value);
  };

  const handleVectorizerSelection = (vectorizer) => {
    setSelectedVectorizer(vectorizer);
  };

  const handleSubmit = async () => {
    try {
      console.log("SUBMIT");
      const inputDataToSend = `${question} ${description} ${previousAction}`; // Combine all three inputs
      const predictResponse = await axios.post(
        "http://127.0.0.1:5000/predict",
        {
          input_data: inputDataToSend,
          // vectorizer: selectedVectorizer, // Include selected vectorizer option
        }
      );
      console.log(predictResponse.data.prediction[0]);
      setPredictions(predictResponse.data.prediction[0]);
  
      const similarResponse = await axios.post(
        "http://127.0.0.1:5000/similar_questions",
        {
          input_data: inputDataToSend,
          vectorizer: selectedVectorizer, // Include selected vectorizer option
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
      <label>Vectorizer:</label> {/* Add label for vectorizer */}
      <div>
        {/* Buttons for vectorizer options */}
        <button 
          onClick={() => handleVectorizerSelection("tfidf+word2vec")}
          style={{marginRight: '10px'}}
          className={selectedVectorizer === "tfidf+word2vec" ? "active" : ""}
        >
          TFIDF + Word2Vec
        </button>
        <button 
          onClick={() => handleVectorizerSelection("universal_sentence_embedding")}
          className={selectedVectorizer === "universal_sentence_embedding" ? "active" : ""}
        >
          Universal Sentence Embedding
        </button>
      </div>
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
