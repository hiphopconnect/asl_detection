import { useState } from "react";
import { ModelName } from "../types/custom_types";

interface PanelProps {
  onButtonClick: (name: ModelName) => void;
}

function AlgorithmSelectPanel({ onButtonClick }: PanelProps) {
  const [isNoneButtonActive, setIsNoneButtonActive] = useState(true);

  const handleButtonClick = (prevButton: string) => {
    if (prevButton === "None") {
      setIsNoneButtonActive(true);
      onButtonClick(ModelName.NONE);
    }
    if (prevButton === "MediaPipeHolistics") {
      setIsNoneButtonActive(false);
      onButtonClick(ModelName.MEDIAPIPE_HOLISTICS);
    }
  };

  return (
    <div className="buttons">
      <button
        className={isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("None")}
      >
        None
      </button>
      <button
        className={!isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("MediaPipeHolistics")}
      >
        MediaPipeHolistics
      </button>
    </div>
  );
}
export default AlgorithmSelectPanel;
