import { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const [similarImages, setSimilarImages] = useState([]); // State to store similar images

  // Handler for when an image is clicked
  const handleImageClick = (index: number) => {
    console.log("Clicked image index:", index);
    fetch('http://localhost:5001/similar-images', { // Make sure the port is correct
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ index })
    })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Log or process the similar images data as needed
        setSimilarImages(data); // Storing similar images data in state
      })
      .catch(error => console.error('Error fetching similar images:', error));
  };

  useEffect(() => {
    fetch("http://localhost:5001/all-images")
      .then(res => res.json())
      .then(data => {
        setImages(data);
        console.log(data);
      });
  }, []);

  return (
    <div className='gallery'>
      {images.length === 0 ? (
        <p>Loading images...</p>
      ) : (
        images.map((image, index) => (
          <div key={image} className='image-container'>
            <img
              src={`http://localhost:5001/images/${image}`}
              alt={`Image ${index}`}
              className='main-img'
              onClick={() => handleImageClick(index)} // Passing the index on click
            />
          </div>
        ))
      )}
      {similarImages.length > 0 && (
        <div className="similar-images">
          <h2>Similar Images</h2>
          {similarImages.map((image, index) => (
            <img key={index} src={`http://localhost:5001/images/${image}`} alt={`Similar Image ${index}`} />
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
