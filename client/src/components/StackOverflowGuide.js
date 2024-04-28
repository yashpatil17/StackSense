import React from 'react';

const StackOverflowGuide = () => {
  return (
    <div className="d-flex w100 ai-center mb16 md:mt0" style={{width: '222%',border: '1px solid #ccc', padding: '10px',  backgroundColor: 'rgba(173, 216, 230, 0.5)' }}>
      <div className="s-notice s-notice__info p24 w70 lg:w100">
        <h2 className="fs-title fw-normal mb8">Writing a good question</h2>
        <p className="fs-body2 mb0">
          You’re ready to <a href="https://stackoverflow.com/help/how-to-ask">ask</a> a <a href="https://stackoverflow.com/help/on-topic">programming-related question</a> and this form will help guide you through the process.
        </p>
        <p className="fs-body2 mt0">
          Looking to ask a non-programming question? See <a href="https://stackexchange.com/sites#technology-traffic">the topics here</a> to find a relevant site.
        </p>
        <h5 className="fw-bold mb8">Steps</h5>
        <ul className="mb0">
          <li>Summarize your problem in a one-line title.</li>
          <li>Describe your problem in more detail.</li>
          <li>Describe what you tried and what you expected to happen.</li>
          <li>Add “tags” which help surface your question to members of the community.</li>
          <li>Review your question and post it to the site.</li>
        </ul>
      </div>
    </div>
  );
};

export default StackOverflowGuide;
