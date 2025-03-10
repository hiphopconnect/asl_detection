import { useEffect, useState } from "react";
import "./App.css";
import AlgorithmSelectPanel from "./components/AlgorithmSelectPanel";
import CameraSelectDropdown from "./components/CameraSelectDropdown";
import LanguageSelectDropdown from "./components/LanguageSelectDropdown";
import UploadButton from "./components/UploadButton";
import VideoDisplay from "./components/VideoDisplay";
import VideoPlayer from "./components/VideoPlayer";
import background from "./newproject.png";
import { ModelName } from "./types/custom_types";

function App() {
  const [activePage, setActivePage] = useState<"live" | "video">("live");

  //State for Camera-ID and the Model
  const [cameraId, setCameraId] = useState<number>(0);
  const [modelName, setModelName] = useState<ModelName>(ModelName.NONE);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string>("");
  const [savedUploadedVideoUrl, setSavedUploadedVideoUrl] =
    useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isGif, setIsGif] = useState<boolean>(false);

  // State for Language
  const [language, setLanguage] = useState<"en" | "de" | "sv">("en");

  // URL of the Livestream
  const liveVideoUrl = `http://127.0.0.1:8000/video/?camera_type=${cameraId}&model_name=${modelName}`;

  // Translation for texts and buttons
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

  // Function to change the language
  const handleLanguageChange = (newLanguage: "en" | "de" | "sv") => {
    setLanguage(newLanguage);
  };

  // handler for Algortihm-Select for the Livestream
  const handleLiveAlgorithmClick = (selectedModel: ModelName) => {
    setModelName(selectedModel);
  };

  // Handler for Algorithm-Select for the Uploaded Video
  const handleUploadAlgorithmClick = async (selectedModel: ModelName) => {
    if (!uploadedVideoUrl) return;

    // set selected Model
    setModelName(selectedModel);

    // save original URL of video
    if (!savedUploadedVideoUrl) {
      setSavedUploadedVideoUrl(uploadedVideoUrl);
    }

    // if no Algortihm was selected, use the original video
    if (selectedModel === ModelName.NONE) {
      setUploadedVideoUrl(savedUploadedVideoUrl);
      setIsGif(false);
      return;
    }

    // show loading indicator
    setIsProcessing(true);

    // ready for backend
    const apiUrl = `http://127.0.0.1:8000/uploaded_video/?video_url=${encodeURIComponent(
      savedUploadedVideoUrl || uploadedVideoUrl
    )}&model_name=${selectedModel}`;

    try {
      // Call BackendAPI
      const response = await fetch(apiUrl);
      const data = await response.json();

      // process answer
      if (data.status === "completed" && data.video_url) {
        // set url to Processed Video
        const serverUrl = "http://127.0.0.1:8000";
        const newVideoUrl = serverUrl + data.video_url;

        // check if the returned value is a gif
        const isGifMedia =
          data.is_gif || newVideoUrl.toLowerCase().includes(".gif");
        setIsGif(isGifMedia);

        // cache breaker to reload the page
        const cacheBreaker = `${newVideoUrl}?t=${Date.now()}`;
        setUploadedVideoUrl(cacheBreaker);
      } else if (data.status === "image_only" && data.image_url) {
        // if just a picture was returned
        const serverUrl = "http://127.0.0.1:8000";
        const imageUrl = serverUrl + data.image_url;

        //set the picture instead of the video
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
      // if there is an error, keep the original video
      setUploadedVideoUrl(savedUploadedVideoUrl);
      setIsGif(false);
    } finally {
      // deactivate Loading screen if the process is done
      setIsProcessing(false);
    }
  };

  // Handle keyboard events for navigation and actions
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        setActivePage(activePage === "live" ? "live" : "video");
        if (activePage === "video") console.log("File upload button triggered");
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activePage]);

  return (
    <div
      className="App"
      style={{
        backgroundImage: `url(${background})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        height: "100vh",
      }}
    >
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
      <LanguageSelectDropdown
        language={language}
        onLanguageChange={handleLanguageChange}
      />
    </div>
  );
}

export default App;
