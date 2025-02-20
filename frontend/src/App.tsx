import { useState } from "react";
import "./App.css";
import VideoDisplay from "./components/VideoDisplay";
import VideoPlayer from "./components/VideoPlayer";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import CameraSelectDropdown from "./components/CameraSelectDropdown";
import UploadButton from "./components/UploadButton";
import { ModelName } from "./types/custom_types";

function App() {
  // activePage controls whether the LIVE feed or the VIDEO page is displayed.
  const [activePage, setActivePage] = useState<"live" | "video">("live");
  const [cameraId, setCameraId] = useState<number>(0);
  const [modelName, setModelName] = useState<ModelName>(ModelName.NONE);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string>("");

  // URL for the live camera feed
  const liveVideoUrl = `http://127.0.0.1:8000/video/?camera_type=${cameraId}&model_name=${modelName}`;

  return (
    <div className="App">
      <div className="centered">
        <div className="feed-and-controls">
          {/* Left side: Video display */}
          {activePage === "live" ? (
            <VideoDisplay streamUrl={liveVideoUrl} />
          ) : (
            <div>
              <h1>Video Feed</h1>
              {uploadedVideoUrl ? (
                <VideoPlayer src={uploadedVideoUrl} />
              ) : (
                <p>No video uploaded yet.</p>
              )}
            </div>
          )}
          {/* Right side: Controls */}
          <div className="controls">
            {/* Navigation buttons */}
            <div className="nav-controls">
              <button
                className={activePage === "live" ? "active" : ""}
                onClick={() => setActivePage("live")}
              >
                LIVE
              </button>
              <button
                className={activePage === "video" ? "active" : ""}
                onClick={() => setActivePage("video")}
              >
                VIDEO
              </button>
            </div>
            {activePage === "live" ? (
              <>
                <h2>Detection</h2>
                <AlgorithmSelectPanel onButtonClick={setModelName} />
                <h2>Camera</h2>
                <CameraSelectDropdown
                  selectedCameraId={cameraId}
                  onCameraSelect={setCameraId}
                />
              </>
            ) : (
              <>
                {/* On the video page, we remove the camera change option */}
                <h2>File Upload</h2>
                <UploadButton onUploadSuccess={setUploadedVideoUrl} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
