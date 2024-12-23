import { useState } from "react";
import { AlgorithmType } from "../types/custom_types";

interface PanelProps {
  onButtonClick: (type: AlgorithmType) => void;
}

function AlgorithmSelectPanel({ onButtonClick }: PanelProps) {
  const [isNoneButtonActive, setIsNoneButtonActive] = useState(true);

  const handleButtonClick = (prevButton: string) => {
    if (prevButton === "None") {
      setIsNoneButtonActive(true);
      onButtonClick(AlgorithmType.NONE);
    } else {
      setIsNoneButtonActive(false);
      onButtonClick(AlgorithmType.DETECTION);
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
        onClick={() => handleButtonClick("Detection")}
      >
        Detection
      </button>
    </div>
  );
}
export default AlgorithmSelectPanel;
