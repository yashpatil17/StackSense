import React, { useState } from "react";
import axios from "axios";
import Navbar from "./components/NavBar";
import "bootstrap/dist/css/bootstrap.min.css"
import PostTitleForm from "./components/PostTitleForm";

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
      <PostTitleForm/>
    </div>
  );
}

export default App;
