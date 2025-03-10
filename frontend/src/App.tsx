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
  const [activePage, setActivePage] = useState<"live" | "video">("live");

  // State für die Kamera-ID und das Modell
  const [cameraId, setCameraId] = useState<number>(0);
  const [modelName, setModelName] = useState<ModelName>(ModelName.NONE);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string>("");
  const [savedUploadedVideoUrl, setSavedUploadedVideoUrl] =
    useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isGif, setIsGif] = useState<boolean>(false);

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
      loading: "Processing video, please wait...",
      noVideo: "No video uploaded yet.",
    },
    de: {
      detection: "Erkennung",
      camera: "Kamera",
      fileUpload: "Datei hochladen",
      live: "LIVE",
      video: "VIDEO",
      loading: "Video wird verarbeitet, bitte warten...",
      noVideo: "Noch kein Video hochgeladen.",
    },
    sv: {
      detection: "Upptäckt",
      camera: "Kamera",
      fileUpload: "Filuppladdning",
      live: "LIVE",
      video: "VIDEO",
      loading: "Bearbetar video, vänta...",
      noVideo: "Inget video uppladdat än.",
    },
  };

  // Funktion, um die Sprache zu ändern
  const handleLanguageChange = (newLanguage: "en" | "de" | "sv") => {
    setLanguage(newLanguage);
  };

  // Handler für Algorithmus-Auswahl im Live-Stream-Modus
  const handleLiveAlgorithmClick = (selectedModel: ModelName) => {
    setModelName(selectedModel);
  };

  // Handler für Algorithmus-Klick im Video-Upload-Modus
  const handleUploadAlgorithmClick = async (selectedModel: ModelName) => {
    if (!uploadedVideoUrl) return;

    // Setze Modell
    setModelName(selectedModel);

    // Speichere Original-URL
    if (!savedUploadedVideoUrl) {
      setSavedUploadedVideoUrl(uploadedVideoUrl);
    }

    // Wenn kein Algorithmus gewählt wurde, zeige Original-Video
    if (selectedModel === ModelName.NONE) {
      setUploadedVideoUrl(savedUploadedVideoUrl);
      setIsGif(false);
      return;
    }

    // Zeige Lade-Indikator
    setIsProcessing(true);

    // Bereite Backend-URL vor
    const apiUrl = `http://127.0.0.1:8000/uploaded_video/?video_url=${encodeURIComponent(
      savedUploadedVideoUrl || uploadedVideoUrl
    )}&model_name=${selectedModel}`;

    try {
      // Rufe Backend-API auf
      const response = await fetch(apiUrl);
      const data = await response.json();

      // Verarbeite Antwort
      if (data.status === "completed" && data.video_url) {
        // Setze URL zum verarbeiteten Video
        const serverUrl = "http://127.0.0.1:8000";
        const newVideoUrl = serverUrl + data.video_url;

        // Prüfe, ob es sich um ein GIF handelt
        const isGifMedia =
          data.is_gif || newVideoUrl.toLowerCase().includes(".gif");
        setIsGif(isGifMedia);

        // Cache-Breaker hinzufügen, damit Browser das neue Video lädt
        const cacheBreaker = `${newVideoUrl}?t=${Date.now()}`;
        setUploadedVideoUrl(cacheBreaker);
      } else if (data.status === "image_only" && data.image_url) {
        // Wenn nur ein Bild zurückgegeben wurde
        const serverUrl = "http://127.0.0.1:8000";
        const imageUrl = serverUrl + data.image_url;

        // Setze das Bild als Video-Ersatz
        setUploadedVideoUrl(imageUrl);
        setIsGif(
          imageUrl.toLowerCase().includes(".gif") ||
            imageUrl.toLowerCase().includes(".jpg")
        );
      } else if (data.status === "error") {
        setUploadedVideoUrl(savedUploadedVideoUrl);
        setIsGif(false);
      }
    } catch (error) {
      // Bei Fehler, behalte die Original-URL
      setUploadedVideoUrl(savedUploadedVideoUrl);
      setIsGif(false);
    } finally {
      // Lade-Indikator ausblenden
      setIsProcessing(false);
    }
  };

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
            <VideoDisplay streamUrl={liveVideoUrl} language={language} />
          ) : (
            <div>
              <h1>{translations[language].fileUpload}</h1>
              {isProcessing ? (
                <div className="processing-indicator">
                  <div className="spinner"></div>
                  <p>{translations[language].loading}</p>
                </div>
              ) : uploadedVideoUrl ? (
                <>
                  {isGif ? (
                    <div className="video-player">
                      <img
                        src={uploadedVideoUrl}
                        alt="Animation"
                        width="640"
                        height="360"
                        style={{
                          display: "block",
                          maxWidth: "100%",
                          objectFit: "contain",
                        }}
                      />
                    </div>
                  ) : (
                    <VideoPlayer src={uploadedVideoUrl} language={language} />
                  )}
                </>
              ) : (
                <p>{translations[language].noVideo}</p>
              )}
            </div>
          )}
          <div className="controls">
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
            {activePage === "live" ? (
              <>
                <h2>{translations[language].detection}</h2>
                <AlgorithmSelectPanel
                  onButtonClick={handleLiveAlgorithmClick}
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
                <h2>{translations[language].fileUpload}</h2>
                <UploadButton
                  onUploadSuccess={setUploadedVideoUrl}
                  language={language}
                />
                {uploadedVideoUrl ? (
                  <>
                    <h2>{translations[language].detection}</h2>
                    <AlgorithmSelectPanel
                      onButtonClick={handleUploadAlgorithmClick}
                      language={language}
                    />
                  </>
                ) : null}
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
