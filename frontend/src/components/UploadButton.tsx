import React, { useRef } from "react";
import "./UploadButton.css";

// Basis-URL für statische Dateien (kann aus einer Umgebungsvariable kommen)
const STATIC_BASE_URL = "http://127.0.0.1:8000/static";

interface UploadButtonProps {
  onUploadSuccess?: (videoUrl: string) => void;
  language: "en" | "de" | "sv"; // Sprachunterstützung hinzufügen
}

// Textübersetzungen je nach Sprache
const translations = {
  en: {
    uploadFile: "Upload File",
    fileExceedsSize: "The file exceeds 500 MB.",
    invalidFileType: "Only MP4 files are allowed.",
    uploadFailed: "Upload failed.",
    uploadError: "Upload error: ",
  },
  de: {
    uploadFile: "Datei hochladen",
    fileExceedsSize: "Die Datei überschreitet 500 MB.",
    invalidFileType: "Nur MP4-Dateien sind erlaubt.",
    uploadFailed: "Upload fehlgeschlagen.",
    uploadError: "Upload Fehler: ",
  },
  sv: {
    uploadFile: "Ladda upp fil",
    fileExceedsSize: "Filen överskrider 500 MB.",
    invalidFileType: "Endast MP4-filer är tillåtna.",
    uploadFailed: "Uppladdning misslyckades.",
    uploadError: "Uppladdningsfel: ",
  },
};

const UploadButton: React.FC<UploadButtonProps> = ({
  onUploadSuccess,
  language,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];

      // Prüfe Dateigröße (max. 500 MB)
      if (file.size > 500 * 1024 * 1024) {
        alert(translations[language].fileExceedsSize);
        return;
      }

      // Prüfe Dateiendung
      if (!file.name.toLowerCase().endsWith(".mp4")) {
        alert(translations[language].invalidFileType);
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
          method: "POST",
          body: formData,
        });

        const responseText = await response.text();

        let data;
        try {
          data = JSON.parse(responseText);
        } catch (jsonError) {
          alert(translations[language].uploadFailed);
          return;
        }

        if (!response.ok) {
          alert(translations[language].uploadError + data.detail);
        } else {
          alert(data.info);
          // Setze die Video-URL generisch zusammen
          if (onUploadSuccess) {
            onUploadSuccess(`${STATIC_BASE_URL}/${data.filename}`);
          }
        }
      } catch (error) {
        alert(translations[language].uploadFailed);
      }
    }
  };

  return (
    <div>
      <button className="upload-button" onClick={handleButtonClick}>
        {translations[language].uploadFile}
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
