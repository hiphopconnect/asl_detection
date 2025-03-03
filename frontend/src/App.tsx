import { useEffect, useState } from "react";
import "./App.css";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import CameraSelectDropdown from "./components/CameraSelectDropdown";
import LanguageSelectDropdown from "./components/LanguageSelectDropdown";
import UploadButton from "./components/UploadButton";
import VideoDisplay from "./components/VideoDisplay";
import VideoPlayer from "./components/VideoPlayer";
import { ModelName } from "./types/custom_types";

function App() {
<<<<<<< frontend/src/App.tsx
  // State für die aktive Seite (LIVE oder VIDEO)
=======
>>>>>>> frontend/src/App.tsx
  const [activePage, setActivePage] = useState<"live" | "video">("live");

  // State für die Kamera-ID und das Modell
  const [cameraId, setCameraId] = useState<number>(0);
  const [modelName, setModelName] = useState<ModelName>(ModelName.NONE);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string>("");

  // State für die Sprache
  const [language, setLanguage] = useState<"en" | "de" | "sv">("en");

  // URL für den Livestream
  const liveVideoUrl = `http://127.0.0.1:8000/video/?camera_type=${cameraId}&model_name=${modelName}`;

  // Übersetzungen für Texte über den Buttons
  const translations = {
    en: {
      detection: "Detection",
      camera: "Camera",
      fileUpload: "File Upload",
      live: "LIVE",
      video: "VIDEO",
    },
    de: {
      detection: "Erkennung",
      camera: "Kamera",
      fileUpload: "Datei hochladen",
      live: "LIVE",
      video: "VIDEO",
    },
    sv: {
      detection: "Upptäckt",
      camera: "Kamera",
      fileUpload: "Filuppladdning",
      live: "LIVE",
      video: "VIDEO",
    },
  };

  // Funktion, um die Sprache zu ändern
  const handleLanguageChange = (newLanguage: "en" | "de" | "sv") => {
    setLanguage(newLanguage);
  };
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
<<<<<<< frontend/src/App.tsx
          {/* Links: Video-Anzeige */}
=======
>>>>>>> frontend/src/App.tsx
          {activePage === "live" ? (
            <VideoDisplay streamUrl={liveVideoUrl} language={language} />
          ) : (
            <div>
              <h1>{translations[language].fileUpload}</h1>
              {uploadedVideoUrl ? (
                <VideoPlayer src={uploadedVideoUrl} language={language} />
              ) : (
                <p>No video uploaded yet.</p>
              )}
            </div>
          )}

<<<<<<< frontend/src/App.tsx
          {/* Rechts: Steuerungselemente */}
          <div className="controls">
            {/* Navigationsbuttons */}
=======
          <div className="controls">
>>>>>>> frontend/src/App.tsx
            <div className="nav-controls">
              <button
                className={activePage === "live" ? "active" : ""}
                onClick={() => setActivePage("live")}
              >
                {translations[language].live}
              </button>
              <button
                className={activePage === "video" ? "active" : ""}
                onClick={() => setActivePage("video")}
              >
                {translations[language].video}
              </button>
            </div>

<<<<<<< frontend/src/App.tsx
            {/* Steuerungen für den Livestream */}
=======
>>>>>>> frontend/src/App.tsx
            {activePage === "live" ? (
              <>
                <h2>{translations[language].detection}</h2>
                <AlgorithmSelectPanel
                  onButtonClick={setModelName}
                  language={language}
                />
                <h2>{translations[language].camera}</h2>
                <CameraSelectDropdown
                  selectedCameraId={cameraId}
                  onCameraSelect={setCameraId}
                  language={language}
                />
              </>
            ) : (
              <>
<<<<<<< frontend/src/App.tsx
                <h2>{translations[language].fileUpload}</h2>
                <UploadButton
                  onUploadSuccess={setUploadedVideoUrl}
                  language={language}
                />
=======
                <h2>File Upload</h2>
                <UploadButton onUploadSuccess={setUploadedVideoUrl} />
>>>>>>> frontend/src/App.tsx
              </>
            )}
          </div>
        </div>
      </div>

      {/* Sprachumschaltung */}
      <LanguageSelectDropdown
        language={language}
        onLanguageChange={handleLanguageChange}
      />
    </div>
  );
}

export default App;
