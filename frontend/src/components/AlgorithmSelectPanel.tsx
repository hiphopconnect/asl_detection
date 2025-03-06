import { useState } from "react";
import { ModelName } from "../types/custom_types";
import "./AlgorithmSelectPanel.css";

interface PanelProps {
  onButtonClick: (name: ModelName) => void;
  language: "en" | "de" | "sv";
}

const translations = {
  en: { none: "None", mediaPipe: "MediaPipeHolistics" },
  de: { none: "Keins", mediaPipe: "MediaPipeHolistics" },
  sv: { none: "Ingen", mediaPipe: "MediaPipeHolistics" },
};

function AlgorithmSelectPanel({ onButtonClick, language }: PanelProps) {
  const [isNoneButtonActive, setIsNoneButtonActive] = useState(true);

  const handleButtonClick = (button: string) => {
    if (button === "None") {
      setIsNoneButtonActive(true);
      onButtonClick(ModelName.NONE);
    }
    if (button === "ASL Fingerspelling") {
      setIsNoneButtonActive(false);
      onButtonClick(ModelName.ASLFINGERSPELLING);
    }
  };

  return (
    <div className="buttons">
      <button
        className={isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("None")}
      >
        {translations[language].none}
      </button>
      <button
        className={!isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("ASL Fingerspelling")}
      >
        ASL Fingerspelling
      </button>
    </div>
  );
}
export default AlgorithmSelectPanel;
