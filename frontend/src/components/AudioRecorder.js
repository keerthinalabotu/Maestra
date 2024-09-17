import React, { useState, useRef } from 'react';
import { Button, HStack, VStack, Text, useToast } from '@chakra-ui/react';
import { MdMic, MdStop } from 'react-icons/md';

const AudioRecorder = ({ onTranscription }) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const toast = useToast();

  React.useEffect(() => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        mediaRecorderRef.current = new MediaRecorder(stream);
        
        mediaRecorderRef.current.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorderRef.current.onstop = sendAudioToBackend;
      })
      .catch(error => {
        console.error('Error accessing microphone:', error);
        toast({
          title: 'Error',
          description: 'Unable to access microphone. Please check your permissions.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      });

    return () => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [toast]);

  const handleStartRecording = () => {
    audioChunksRef.current = [];
    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  const sendAudioToBackend = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      onTranscription(data.transcription); // Pass transcription to parent component
    } catch (error) {
      console.error('Error transcribing audio:', error);
      toast({
        title: 'Transcription Error',
        description: 'Failed to transcribe audio. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <VStack spacing={4}>
      <HStack spacing={4}>
        <Button
          leftIcon={<MdMic />}
          onClick={handleStartRecording}
          colorScheme="green"
          isDisabled={isRecording}
        >
          Start Recording
        </Button>
        <Button
          leftIcon={<MdStop />}
          onClick={handleStopRecording}
          colorScheme="red"
          isDisabled={!isRecording}
        >
          Stop Recording
        </Button>
      </HStack>
      {isRecording && <Text>Recording...</Text>}
    </VStack>
  );
};

export default AudioRecorder;
