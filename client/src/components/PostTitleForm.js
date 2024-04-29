import React, { useState } from "react";
import { Container, Form, Button, Spinner } from "react-bootstrap";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import Question from "./Question";
import StackOverflowGuide from "./StackOverflowGuide";
import PredictionsList from "./tag";
import QuestionBox from "./QuestionBox";

const MyForm = () => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [pastActions, setPastActions] = useState("");
  const [selectedVectorizer, setSelectedVectorizer] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [similar, setSimilar] = useState([]);
  const [similarBody, setSimilarBody] = useState([]);

  const [numQuestions, setNumQuestions] = useState(20); // State variable for number of questions
  const [loading, setLoading] = useState(false);

  // Your existing code for fetching similar questions and rendering them

  // Function to handle slider change
  const handleSliderChange = (event) => {
    setNumQuestions(parseInt(event.target.value)); // Update number of questions to display
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      console.log("Submitting form...");
      const predictResponse = await axios.post(
        "http://127.0.0.1:5000/predict",
        {
          input_data: `${title} ${description} ${pastActions}`,
        }
      );

      setPredictions(predictResponse.data.prediction[0]);

      //   const similarResponse = await axios.post('http://127.0.0.1:5000/similar_questions', {
      //     input_data: `${title} ${description} ${pastActions}`,
      //     vectorizer: selectedVectorizer,
      //   });

      //   setSimilar(similarResponse.data.similar_questions);
      // } catch (error) {
      //   console.error('Error:', error);
      // }
      const inputDataToSend = `${title} ${description} ${pastActions}`; // Combine all three inputs
      setLoading(true);
      const similarResponse = await axios.post(
        "http://127.0.0.1:5000/similar_questions",
        {
          input_data: inputDataToSend,
          vectorizer: selectedVectorizer, // Include selected vectorizer option
        }
      );
      // console.log(similarResponse.data.similar_questions);
      // setSimilar(similarResponse.data.similar_questions);
      console.log(similarResponse.data);
      setSimilar(similarResponse.data.title);
      setSimilarBody(similarResponse.data.body);
      setLoading(false);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleVectorizerSelection = (vectorizer) => {
    setSelectedVectorizer(vectorizer);
  };
  // const Spinner = () => (
  //   <div className="spinner" style={{ textAlign: 'center', marginTop: '20px' }}>
  //     Loading...
  //   </div>
  // );

  return (
    <div style={{ marginBottom: "50px" }}>
      <Container
        style={{
          marginTop: "100px",
          marginBottom: "50px",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <div style={{ width: "45%" }}>
          <div style={{ marginBottom: "20px" }}>
            <Question />
          </div>
          <div style={{ marginBottom: "20px" }}>
            <StackOverflowGuide />
          </div>
        </div>
      </Container>
      <Container
        style={{
          margin: "100 100 20 20",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <div style={{ width: "45%" }}>
          {" "}
          {/* Adjust width as needed */}
          {/* <div style={{ marginBottom: '20px' }}>
          <Question />
        </div>
        <div style={{ marginBottom: '20px' }}>
          <StackOverflowGuide />
        </div> */}
          <Form onSubmit={handleSubmit}>
            <Form.Group controlId="formTitle">
              <Form.Label>Title</Form.Label>
              <Form.Control
                type="text"
                placeholder="Enter title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                style={{ width: "50%", marginTop: "5px", marginBottom: "15px" }}
              />
            </Form.Group>

            <Form.Group controlId="formDescription">
              <Form.Label>Description</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                placeholder="Enter description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                style={{ width: "50%", marginTop: "5px", marginBottom: "15px" }}
              />
            </Form.Group>

            <Form.Group controlId="formPastActions">
              <Form.Label>
                What did you try and what were you expecting?
              </Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                placeholder="Enter past actions"
                value={pastActions}
                onChange={(e) => setPastActions(e.target.value)}
                style={{ width: "50%", marginTop: "5px", marginBottom: "15px" }}
              />
            </Form.Group>

            <div>
              <Button
                variant={
                  selectedVectorizer === "tfidf+word2vec"
                    ? "primary"
                    : "outline-primary"
                }
                onClick={() => handleVectorizerSelection("tfidf+word2vec")}
                style={{ marginRight: "10px" }}
              >
                TFIDF + Word2Vec
              </Button>
              <Button
                variant={
                  selectedVectorizer === "universal_sentence_embedding"
                    ? "primary"
                    : "outline-primary"
                }
                onClick={() =>
                  handleVectorizerSelection("universal_sentence_embedding")
                }
              >
                Universal Sentence Embedding
              </Button>
            </div>

            <Button
              variant="primary"
              type="submit"
              style={{ marginTop: "15px", marginBottom: "15px" }}
            >
              Submit
            </Button>
          </Form>
          <PredictionsList predictions={predictions} />{" "}
          {/* Moved PredictionsList to here */}
        </div>
        {/* {
        similar?.length !== 0 && <div style={{ width: '45%', border: '2px solid blue', padding: '10px' }}>
        <h2 style={{ textAlign: 'center' }}>Similar Questions:</h2>
        <ol>
          {similar?.map((question, index) => (
            <li key={index}>
              <QuestionBox question={question} similarBody={similarBody} index={index} />
            </li>
          ))}
        </ol>
      </div>
      } */}
        {/* {loading ? (
        <div className="spinner" style={{ alignItems: 'center', marginRight: '200px', display: 'flex', justifyContent: 'center' }}>
          <Spinner />
        </div>
      ) : (
        similar.length !== 0 && (
          <div style={{ width: '45%', border: '2px solid #0d6efd', padding: '10px', borderRadius: '10px' }}>
            <h2 style={{ textAlign: 'center' }}>Similar Questions:</h2>
            <ol>
              {similar.map((question, index) => (
                <li key={index}>
                  <QuestionBox question={question} similarBody={similarBody} index={index} />
                </li>
              ))}
            </ol>
          </div>
        )
      )} */}

        {loading ? (
          <div
            className="spinner"
            style={{
              alignItems: "center",
              marginRight: "200px",
              display: "flex",
              justifyContent: "center",
            }}
          >
            <Spinner />
          </div>
        ) : (
          similar.length !== 0 && (
            <div
              style={{
                width: "45%",
                border: "2px solid #0d6efd",
                padding: "10px",
                borderRadius: "10px",
              }}
            >
              {/* Slider component */}
              <div style={{ textAlign: "center" }}>
                {/* Slider component */}
                <input
                  type="range"
                  min="1"
                  max={similar.length}
                  value={numQuestions}
                  onChange={handleSliderChange}
                />
              </div>
              <p style={{ textAlign: "center" }}>
                Displaying {numQuestions} out of {similar.length} questions
              </p>
              <h2 style={{ textAlign: "center" }}>Similar Questions:</h2>
              <ol>
                {similar.slice(0, numQuestions).map(
                  (
                    question,
                    index // Display only sliced questions
                  ) => (
                    <li key={index}>
                      <QuestionBox
                        question={question}
                        similarBody={similarBody}
                        index={index}
                      />
                    </li>
                  )
                )}
              </ol>
            </div>
          )
        )}
      </Container>
    </div>
  );
};

export default MyForm;
