import React, { useRef } from "react";
import "./VideoPlayer.css";

interface VideoPlayerProps {
  src: string;
  language: "en" | "de" | "sv";
}

const translations = {
  en: { play: "Play", pause: "Pause", stop: "Stop", translate: "Translate" },
  de: {
    play: "Abspielen",
    pause: "Pause",
    stop: "Stopp",
    translate: "Übersetzen",
  },
  sv: { play: "Spela", pause: "Paus", stop: "Stopp", translate: "Översätt" },
};

const VideoPlayer: React.FC<VideoPlayerProps> = ({ src, language }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const handlePlay = () => {
    videoRef.current?.play();
  };

  const handlePause = () => {
    videoRef.current?.pause();
  };

  const handleStop = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  return (
    <div className="video-player">
      <video
        ref={videoRef}
        src={src}
        width="640"
        height="360"
        controls={false}
      />
      <div className="player-controls">
        <button onClick={handlePlay}>{translations[language].play}</button>
        <button onClick={handlePause}>{translations[language].pause}</button>
        <button onClick={handleStop}>{translations[language].stop}</button>
        <button>{translations[language].translate}</button>
      </div>
    </div>
  );
};

export default VideoPlayer;
