import React from "react";
import "./CameraSelectDropdown.css";

interface CameraOption {
  id: number;
  name: string;
}

interface CameraSelectDropdownProps {
  selectedCameraId: number;
  onCameraSelect: (camera: number) => void;
}

// Hardcoded Options: z.B. Kamera 0 und Kamera 4
const hardcodedCameraOptions: CameraOption[] = [
  { id: 0, name: "Camera 0" },
  { id: 4, name: "Camera 4" },
];

const CameraSelectDropdown: React.FC<CameraSelectDropdownProps> = ({
  selectedCameraId,
  onCameraSelect,
}) => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedId = parseInt(event.target.value, 10);
    onCameraSelect(selectedId);
  };

  return (
    <select
      className="camera-dropdown"
      value={selectedCameraId}
      onChange={handleChange}
    >
      {hardcodedCameraOptions.map((option) => (
        <option key={option.id} value={option.id}>
          {option.name}
        </option>
      ))}
    </select>
  );
};

export default CameraSelectDropdown;
