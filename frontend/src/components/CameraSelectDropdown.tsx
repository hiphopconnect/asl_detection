import React from "react";
import "./CameraSelectDropdown.css";

interface CameraOption {
  id: number;
  name: string;
}

interface CameraSelectDropdownProps {
  selectedCameraId: number;
  onCameraSelect: (camera: number) => void;
  language: "en" | "de" | "sv";
}

const translations = {
  en: { camera0: "Camera 0", camera4: "Camera 4" },
  de: { camera0: "Kamera 0", camera4: "Kamera 4" },
  sv: { camera0: "Kamera 0", camera4: "Kamera 4" },
};

const CameraSelectDropdown: React.FC<CameraSelectDropdownProps> = ({
  selectedCameraId,
  onCameraSelect,
  language,
}) => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedId = parseInt(event.target.value, 10);
    onCameraSelect(selectedId);
  };

  const cameraOptions: CameraOption[] = [
    { id: 0, name: translations[language].camera0 },
    { id: 4, name: translations[language].camera4 },
  ];

  return (
    <select
      className="camera-dropdown"
      value={selectedCameraId}
      onChange={handleChange}
    >
      {cameraOptions.map((option) => (
        <option key={option.id} value={option.id}>
          {option.name}
        </option>
      ))}
    </select>
  );
};

export default CameraSelectDropdown;
