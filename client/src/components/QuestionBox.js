import React from 'react';

const QuestionBox = ({ question }) => {
  return (
    <div style={{ padding: '10px', marginBottom: '20px' }}>
      {question}
    </div>
  );
};

export default QuestionBox;
