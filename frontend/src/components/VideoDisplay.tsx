import "./VideoDisplay.css";

interface Props {
  streamUrl: string;
  language: "en" | "de" | "sv";
}

const translations = {
  en: "No input available...",
  de: "Kein Eingabegerät verfügbar...",
  sv: "Ingen ingång tillgänglig...",
};

function VideoDisplay({ streamUrl, language }: Props) {
  return (
    <div>
      <img src={streamUrl} alt={translations[language]} />
    </div>
  );
}

export default VideoDisplay;
