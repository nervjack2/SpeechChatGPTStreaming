// AudioStreamer.js
import React, { useState, useEffect } from 'react';
import { Button } from '@mui/material';

const AudioStreamer = ({ socket }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  useEffect(() => {
    // Check for support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error('Media Devices not supported by the browser.');
      return;
    }

    // Get user media
    const getMedia = async () => {
      try {
        console.log('Requesting microphone access...');
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('Microphone access granted:', stream);
        
        const recorder = new MediaRecorder(stream);
        setMediaRecorder(recorder);

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0 && socket) {
            socket.emit('audio', event.data);
          }
        };

        recorder.onerror = (event) => {
          console.error('MediaRecorder error:', event);
        };
      } catch (error) {
        console.error('Failed to get user media:', error);
      }
    };

    getMedia();
    return () => mediaRecorder?.stream.getTracks().forEach(track => track.stop());
  }, [socket]);

  const startRecording = () => {
    if (!mediaRecorder) return;
    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (!mediaRecorder) return;
    mediaRecorder.stop();
    setIsRecording(false);
  };

  return (
    <div>
      <Button variant="contained" onClick={startRecording} disabled={isRecording}>
        Start Recording
      </Button>
      <Button variant="contained" onClick={stopRecording} disabled={!isRecording}>
        Stop Recording
      </Button>
    </div>
  );
};

export default AudioStreamer;
