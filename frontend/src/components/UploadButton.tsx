import React, { useRef } from "react";
import "./UploadButton.css";

interface UploadButtonProps {
  onFileSelected?: (file: File) => void;
}

const UploadButton: React.FC<UploadButtonProps> = ({ onFileSelected }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];

      // Optional: Check file size (max 500 MB)
      if (file.size > 500 * 1024 * 1024) {
        alert("The file exceeds 500 MB.");
        return;
      }
      // Optional: Check file extension
      if (!file.name.toLowerCase().endsWith(".mp4")) {
        alert("Only MP4 files are allowed.");
        return;
      }

      // Create FormData and send the file to the backend endpoint
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          const data = await response.json();
          alert("Upload error: " + data.detail);
        } else {
          const data = await response.json();
          alert(data.info);
          if (onFileSelected) onFileSelected(file);
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
