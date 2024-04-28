import React, { useState } from 'react';

const QuestionBox = ({ question, similarBody, index }) => {
  const [showBody, setShowBody] = useState(false);

  const toggleBody = () => {
    setShowBody(!showBody);
  };

  const removeTags = (html) => {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    return doc.body.textContent || '';
  };

  return (
    <div style={{ padding: '10px', marginBottom: '20px', width: '100%' }}>
      <div style={{ wordWrap: 'break-word' }}>{question}</div>
      <button onClick={toggleBody}>Show Body</button>
      {showBody && (
        <div
          style={{
            marginTop: '10px',
            border: '1px solid black',
            padding: '5px',
            wordWrap: 'break-word',
            maxWidth: '100%'
          }}
        >
          {removeTags(similarBody[index])}
        </div>
      )}
    </div>
  );
};

export default QuestionBox;
