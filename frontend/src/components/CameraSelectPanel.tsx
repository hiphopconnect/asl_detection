import { useState } from "react";
import { CameraType } from "../types/custom_types";
import "./CameraSelectPanel.css";

interface PanelProps {
  onButtonClick: (type: CameraType) => void;
}

function CameraSelectPanel({ onButtonClick }: PanelProps) {
  const [isRGBButtonActive, setIsRGBButtonActive] = useState(true);

  const handleButtonClick = (prevButton: string) => {
    if (prevButton === "rgb") {
      setIsRGBButtonActive(true);
      onButtonClick(CameraType.RGB);
    } else {
      setIsRGBButtonActive(false);
      onButtonClick(CameraType.IR);
    }
  };

  return (
    <div className="buttons">
      <button
        className={isRGBButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("rgb")}
      >
        RGB Camera
      </button>
      <button
        className={!isRGBButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("ir")}
      >
        IR Camera
      </button>
    </div>
  );
}

export default CameraSelectPanel;
