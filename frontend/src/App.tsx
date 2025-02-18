import { useState } from "react";
import "./App.css";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import VideoDisplay from "./components/VideoDisplay";
import UploadButton from "./components/UploadButton";
import CameraSelectDropdown from "./components/CameraSelectDropdown";
import { ModelName } from "./types/custom_types";

function App() {
  const [cameraId, setCameraId] = useState<number>(0);
  const [modelName, setModelName] = useState<ModelName>(ModelName.NONE);
  const videoUrl: string = "http://127.0.0.1:8000/video/";

  const handleFileSelected = (file: File) => {
    console.log("File selected:", file);
  };

  return (
    <div className="App">
      <div className="centered">
        <div className="feed-and-controls">
          {/* Video left */}
          <VideoDisplay
            streamUrl={`${videoUrl}?camera_type=${cameraId}&model_name=${modelName}`}
          />
          {/* Controls right */}
          <div className="controls">
            <h2>Detection</h2>
            <AlgorithmSelectPanel onButtonClick={setModelName} />

            <h2>Camera</h2>
            <CameraSelectDropdown
              selectedCameraId={cameraId}
              onCameraSelect={setCameraId}
            />
            <h2>File Upload</h2>
             <UploadButton onFileSelected={handleFileSelected} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
