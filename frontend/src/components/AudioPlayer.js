import React, { useState, useEffect } from 'react';

const AudioPlayer = ({ audioFile }) => {
  const [audio, setAudio] = useState(null);

  useEffect(() => {
    if (audioFile) {
      setAudio(new Audio(audioFile));
    }
  }, [audioFile]);

  const playAudio = () => {
    if (audio) {
      audio.play();
    }
  };

  return (
    <div>
      <button onClick={playAudio}>Play Response</button>
    </div>
  );
};

export default AudioPlayer;