import { useEffect, useState } from 'react'
import './App.css'

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
          <img key={i} src={`http://localhost:5001/images/${image}`} alt={`Image ${i}`} className='main-img'  />
        ))
      )}
    </div>
  )
}

export default App
