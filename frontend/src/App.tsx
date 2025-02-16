import { useState } from "react";
import "./App.css";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import VideoDisplay from "./components/VideoDisplay";
import { AlgorithmType, CameraType } from "./types/custom_types";

function App() {
  const [algorithmType, setAlgorithmType] = useState<AlgorithmType>(
    AlgorithmType.NONE
  );
  // Fester Kameratyp, da der CameraSelectPanel entfernt wurde:
  const cameraType = CameraType.RGB;
  const videoUrl: string = "http://127.0.0.1:8000/video/";

  return (
    <div className="App">
      <div className="centered">
        <div className="feed-and-controls">
          <VideoDisplay
            streamUrl={`${videoUrl}?camera_type=${cameraType}&algorithm_type=${algorithmType}`}
          />
          <div className="controls">
            <AlgorithmSelectPanel onButtonClick={setAlgorithmType} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
