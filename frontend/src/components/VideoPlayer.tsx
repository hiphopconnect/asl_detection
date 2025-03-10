import React, { useEffect, useRef } from "react";
import "./VideoPlayer.css";

interface VideoPlayerProps {
  src: string;
  language: "en" | "de" | "sv";
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ src, language }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Lade das Video neu, wenn sich die Quelle ändert
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load();
    }
  }, [src]);

  return (
    <div className="video-player">
      <video
        ref={videoRef}
        width="640"
        height="360"
        controls={true}
        preload="auto"
      >
        <source src={src} type="video/mp4" />
        Ihr Browser unterstützt dieses Video nicht.
      </video>
    </div>
  );
};

export default VideoPlayer;
