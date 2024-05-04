import React, { useState, useEffect } from 'react';

function App() {
  const [images, setImages] = useState([]); // This will hold image filenames

  useEffect(() => {
    fetch("http://localhost:5001/all-images").then(
      res => res.json()
    ).then(
      data => {
        setImages(data); // Store the filenames
        console.log(data);
      }
    )
  }, []);

  return (
    <div>
      {images.length === 0 ? (
        <p>Loading images...</p>
      ) : (
        images.map((image, i) => (
          <img key={i} src={`http://localhost:5001/images/${image}`} alt={`Image ${i}`} style={{ margin: '10px', height: '100px' }} />
        ))
      )}
    </div>
  )
}

export default App;
