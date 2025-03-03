import { useEffect, useState } from "react";
import "./App.css";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import CameraSelectDropdown from "./components/CameraSelectDropdown";
import UploadButton from "./components/UploadButton";
import VideoDisplay from "./components/VideoDisplay";
import VideoPlayer from "./components/VideoPlayer";
import { ModelName } from "./types/custom_types";

function App() {
  const [activePage, setActivePage] = useState<"live" | "video">("live");
  const [cameraId, setCameraId] = useState<number>(0);
  const [modelName, setModelName] = useState<ModelName>(ModelName.NONE);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string>("");

  const liveVideoUrl = `http://127.0.0.1:8000/video/?camera_type=${cameraId}&model_name=${modelName}`;

  const buttons = ["LIVE", "VIDEO"];

  // Handle keyboard events for navigation and actions
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        if (activePage === "live") {
          // Perform the action for "LIVE" page
          setActivePage("live");
        } else if (activePage === "video") {
          // Perform the action for "VIDEO" page
          // In "VIDEO" page, trigger file upload action
          setActivePage("video");
          // Optional: Here you can trigger the upload button action directly if needed
          console.log("File upload button triggered");
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    // Cleanup the event listener on unmount
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activePage]); // Add activePage as a dependency

  return (
    <div className="App">
      <div className="centered">
        <div className="feed-and-controls">
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

          <div className="controls">
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
