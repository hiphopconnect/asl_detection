import React, { useRef } from "react";
import "./UploadButton.css";

// Basis-URL für statische Dateien (kann aus einer Umgebungsvariable kommen)
const STATIC_BASE_URL = "http://127.0.0.1:8000/static";

interface UploadButtonProps {
  onUploadSuccess?: (videoUrl: string) => void;
}

const UploadButton: React.FC<UploadButtonProps> = ({ onUploadSuccess }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];

      // Prüfe Dateigröße (max. 500 MB)
      if (file.size > 500 * 1024 * 1024) {
        alert("The file exceeds 500 MB.");
        return;
      }
      // Prüfe Dateiendung
      if (!file.name.toLowerCase().endsWith(".mp4")) {
        alert("Only MP4 files are allowed.");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
          method: "POST",
          body: formData,
        });
        console.log("Response status:", response.status);

        const responseText = await response.text();
        console.log("Raw response text:", responseText);

        let data;
        try {
          data = JSON.parse(responseText);
          console.log("Parsed response data:", data);
        } catch (jsonError) {
          console.error("JSON parse error:", jsonError);
          alert("Upload failed due to invalid response format.");
          return;
        }

        if (!response.ok) {
          alert("Upload error: " + data.detail);
        } else {
          alert(data.info);
          // Setze die Video-URL generisch zusammen
          if (onUploadSuccess) {
            onUploadSuccess(`${STATIC_BASE_URL}/${data.filename}`);
          }
        }
      } catch (error) {
        console.error("Upload error:", error);
        alert("Upload failed.");
      }
    }
  };

  return (
    <div>
      <button className="upload-button" onClick={handleButtonClick}>
        Upload File
      </button>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileChange}
        accept="video/mp4"
      />
    </div>
  );
};

export default UploadButton;
