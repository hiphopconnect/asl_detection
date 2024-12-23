import { useState } from "react";
import "./App.css";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import CameraSelectPanel from "./components/CameraSelectPanel";
import VideoDisplay from "./components/VideoDisplay";
import { AlgorithmType, CameraType } from "./types/custom_types";

function App() {
  const [cameraType, setCameraType] = useState<CameraType>(CameraType.RGB);
  const [algorithmType, setAlgorithmType] = useState<AlgorithmType>(
    AlgorithmType.NONE
  );
  const videoUrl: string = "http://127.0.0.1:8000/video/";

  return (
    <div className="App">
      <div className="centered">
        <div
          // Debug handler to log the changes in backend endpoint configuration
          onClick={() =>
            console.log(
              videoUrl +
                `?camera_type=${cameraType}&algorithm_type=${algorithmType}`
            )
          }
        >
          <AlgorithmSelectPanel onButtonClick={setAlgorithmType} />
          <VideoDisplay
            streamUrl={
              videoUrl +
              `?camera_type=${cameraType}&algorithm_type=${algorithmType}`
            }
          />
          <CameraSelectPanel onButtonClick={setCameraType} />
        </div>
      </div>
    </div>
  );
}

export default App;
