import React from 'react';

const PredictionsList = ({ predictions }) => {
  return (
    <div>
      <h2>Recommended Tags:</h2>
      <div style={{ display: 'flex', flexWrap: 'wrap' }}>
        {predictions.map((prediction, index) => (
          <div key={index} style={{ backgroundColor: 'white', padding: '5px', margin: '5px', borderRadius: '5px', color: '#0d6efd', border: '1px solid #0d6efd' }}>
            {prediction}
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionsList;
