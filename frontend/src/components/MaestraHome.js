
import React, { useState, useEffect, useRef } from 'react';
import { Box, Heading, Button, Text, VStack, HStack, Grid, GridItem, Input, FormControl, FormLabel, useMediaQuery, useToast } from '@chakra-ui/react';
import { MdMic, MdFileUpload } from 'react-icons/md';
import Recorder from 'recorder-js';

const AudioWave = ({ isSpeaking }) => {
  const [waveform, setWaveform] = useState([]);

  useEffect(() => {
    let interval;
    if (isSpeaking) {
      interval = setInterval(() => {
        setWaveform(Array.from({ length: 50 }, () => Math.random() * 40));
      }, 100);
    } else {
      setWaveform([]); // Clear the waveform when not speaking
    }
    return () => clearInterval(interval);
  }, [isSpeaking]);

  return (
    <HStack spacing={1} h="20px">
      {waveform.map((height, index) => (
        <Box key={index} w="2px" bg="blue.500" h={`${height}px`} />
      ))}
    </HStack>
  );
};

const MaestraHome = ({ initialResponse }) => {
  const [maestraQuestion, setMaestraQuestion] = useState('');
  const [userAnswer, setUserAnswer] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [topic, setTopic] = useState('');
  const [field, setField] = useState('');
  const [file, setFile] = useState(null);
  const [isFirstPlay, setIsFirstPlay] = useState(true);
  const audioPlayerRef = useRef(null);

  const [isLargerThan1280] = useMediaQuery("(min-width: 1280px)");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const toast = useToast();
  // const location = useLocation();
  // const { extractedResponse } = location.state || {}; 

  console.log(initialResponse);

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  let recorder;

  // useEffect(() => {
  //   // Request permission to access audio when the component mounts
  //   navigator.mediaDevices.getUserMedia({ audio: true })
  //   .then(stream => {
  //     const input = audioContext.createMediaStreamSource(stream);

  //     // Initialize the recorder with the audio input stream
  //     recorder = new Recorder(audioContext, {
  //       type: 'audio/wav'
  //     });
  //     recorder.init(stream);
  //   })
  //   .catch(error => {
  //     console.error('Error accessing microphone:', error);
  //     alert('Unable to access microphone. Please check your permissions.');
  //   });

  //   return () => {
  //     // Stop audio context when component unmounts
  //     if (recorder) {
  //       recorder.close();
  //     }
  //   };
  // }, []);

  useEffect(() => {
    // Request permission to access audio when the component mounts
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
  }, []);

  // const handleStartRecording = () => {
  //   if (recorder) {
  //     recorder.start()
  //       .then(() => {
  //         setIsRecording(true);
  //         setIsSpeaking(true);
  //       })
  //       .catch(error => console.error('Error starting recording:', error));
  //   }
  // };

  // const handleStopRecording = () => {
  //   if (recorder) {
  //     recorder.stop()
  //       .then(({ blob }) => {
  //         console.log('Audio Blob:', blob);
  //         sendAudioToBackend(blob);
  //       })
  //       .catch(error => console.error('Error stopping recording:', error));

  //     setIsRecording(false);
  //     setIsSpeaking(false);
  //   }
  // };

  const handleStartRecording = () => {
    audioChunksRef.current = [];
    mediaRecorderRef.current.start();
    setIsRecording(true);
    setIsSpeaking(true);
  };

  const handleStopRecording = () => {
    mediaRecorderRef.current.stop();
    // sendAudioToBackend(blob);
    setIsRecording(false);
    setIsSpeaking(false);
  };

  // const sendAudioToBackend = async (audioBlob) => {
  //   const formData = new FormData();
  //   formData.append('file', audioBlob, 'recording.wav'); // Name the file with extension
  //   console.log(file);
  //   try {
  //     const response = await fetch('http://localhost:8000/transcribe', {
  //       method: 'POST',
  //       body: formData,
  //     });
  //     const data = await response.json();

  //     console.log('Transcription Response:', data);
  //     setTranscript(prev => [...prev, { speaker: 'You', text: data.transcription }]);
  //   } catch (error) {
  //     console.error('Error transcribing audio:', error);
  //     alert('Failed to transcribe audio. Please try again.');
  //   }
  // };

  const sendAudioToBackend = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    console.log('Audio Blob:', audioBlob);
    console.log('Audio Blob Type:', audioBlob.type);
    console.log('Audio Blob Size:', audioBlob.size);
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    console.log(formData.name)


    for (let [key, value] of formData.entries()) {
      console.log(`${key}:`, value);
      if (value instanceof File) {
        console.log(`${key} details:`, {
          name: value.name,
          type: value.type,
          size: value.size
        });
      }
    }

    // for (let [key, value] of formData.entries()) {
    //   console.log(`${key}: ${value}`);
    // }

    try {
      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      console.log('Response body:', data);
      setTranscript(prev => [...prev, { speaker: 'You', text: data.transcription }]);
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


  const handlePlayMaestraQuestion = async () => {
    console.log(initialResponse)
    if (isFirstPlay) {
        try {
          const response = await fetch('http://localhost:8000/conversation_initial', {
            method: 'POST', // Explicitly setting the method to POST
            headers: {
              'Content-Type': 'application/json', // Set content type to JSON
            },
            body: JSON.stringify({ extracted_response: initialResponse }),
        });

        console.log('Response status:', response.status); 
        // const responseBody = await response.text();
        // console.log('Response body:', responseBody); 
        // console.log('ERMMM'); 
        
        // const data_temp = await response.json();
        // if (!response.ok) throw new Error(data.message);

        console.log('what is going on bro'); 
        const data = await response.json();
        console.log(data);

        if (!response.ok) {
          console.log('NO OKKK:'); 
          throw new Error('Failed to fetch Maestra question');

        }
        
        // setAudioUrl(data.audio_file);
        
        // setMaestraQuestion(data.extracted_response);
        console.log('Audio url:', data.audio_file); 
        // relative_path = os.path.relpath(data.audio_url, "static")
        // relative_path = relative_path.replace("\\", "/") 

        let relative_path = data.audio_file.replace(/\\/g, '/');
        console.log('relative url:', relative_path); 

        const audio_url=relative_path
        // const audio_url = `http://localhost:8000/static/${relative_path}`;
        console.log('Audio url:', audio_url); 
        
        const audio = new Audio(audio_url); // Assuming the backend returns an audio URL



        // setTranscript(prev => [...prev, { speaker: 'Maestra', text: initialResponse }]);
        
        // Create and play audio
        // const audio = new Audio(`http://localhost:8000${data.audio_url}`); // Assuming the backend returns an audio URL
        audio.onplay = () => setIsSpeaking(true);
        audio.onended = () => setIsSpeaking(false);
        // audio.play();

        audio.play().catch(error => {
          console.error('Error playing audio:', error);
        });
  
        
        // Save audio player reference for future use
        audioPlayerRef.current = audio;
        setIsFirstPlay(false);

        setTranscript(prev => [...prev, { speaker: 'Maestra', text: initialResponse }]);
      } catch (error) {
        console.error('Error getting Maestra question:', error);
        toast({
          title: 'Error',
          description: 'Failed to get Maestra\'s question. Please try again.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    } else {
      // For subsequent plays, use the saved audio player
      if (audioPlayerRef.current) {
        audioPlayerRef.current.play();
      }
    }


  };

  const handlePlayAudio = () => {
    const audio = new Audio(audioUrl);
    audio.play().catch(error => {
      console.error('Error playing audio:', error);
    });
    setIsPlaying(true);
    audio.onended = () => setIsPlaying(false);
  };

  const handleFileUpload = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmitFile = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('topic', topic);
    formData.append('field', field);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log('File uploaded successfully:', data);
      // Handle success (e.g., show a success message)
    } catch (error) {
      console.error('Error uploading file:', error);
      // Handle error (e.g., show an error message)
    }
  };



  return (
    <Box maxW="100%" mx="auto" p={[4, 6]} minH="100vh" display="flex" flexDirection="column" bg="gray.800" color="white">
      <Grid 
        templateColumns={isLargerThan1280 ? "1fr 1fr" : "1fr"} 
        gap={6} 
        flex="1"
      >
        <GridItem colSpan={1} rowSpan={1}>
          <Box borderWidth={1} borderRadius="lg" boxShadow="lg" p={6} h="100%">
            <Heading as="h1" size="xl" color="blue.600" mb={4}>
              Maestra
            </Heading>
            <AudioWave isSpeaking={isSpeaking} />
            <VStack spacing={4} align="stretch" mt={4}>
              <Box>
                <Text mb={2}>Maestra's question:</Text>
                <Text>{maestraQuestion}</Text>
                <Button onClick={handlePlayMaestraQuestion} mt={2} colorScheme="blue">
                  Play Question
                </Button>
              </Box>
              <Box>
                <Text mb={2}>Your answer:</Text>
                <HStack>
                  <Button
                    leftIcon={<MdMic />}
                    onClick={isRecording ? handleStopRecording : handleStartRecording}
                    colorScheme={isRecording ? "red" : "green"}
                  >
                    {isRecording ? 'Stop recording' : 'Start recording'}
                  </Button>
                </HStack>
              </Box>
            </VStack>
          </Box>
        </GridItem>
        <GridItem colSpan={1} rowSpan={isLargerThan1280 ? 2 : 1}>
          <Box borderWidth={1} borderRadius="lg" boxShadow="lg" p={6} h="100%">
            <Heading as="h2" size="lg" mb={4}>
              Transcript
            </Heading>
            <VStack align="stretch" spacing={2} overflowY="auto" h="calc(100% - 40px)">
              {transcript.map((entry, index) => (
                <Text key={index}>
                  <strong>{entry.speaker}:</strong> {entry.text}
                </Text>
              ))}
            </VStack>
          </Box>
        </GridItem>
      </Grid>
      {/* <Box mt={6} borderWidth={1} borderRadius="lg" boxShadow="lg" p={6}>
        <Heading as="h2" size="lg" mb={4}>
          Upload File
        </Heading>
        <Grid templateColumns={["1fr", "1fr 1fr"]} gap={4}>
          <GridItem>
            <FormControl>
              <FormLabel>Topic</FormLabel>
              <Input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Enter topic" />
            </FormControl>
          </GridItem>
          <GridItem>
            <FormControl>
              <FormLabel>Field</FormLabel>
              <Input value={field} onChange={(e) => setField(e.target.value)} placeholder="Enter field" />
            </FormControl>
          </GridItem>
        </Grid>
        <FormControl mt={4}>
          <FormLabel>File</FormLabel>
          <Input type="file" onChange={handleFileUpload} accept=".pdf,.doc,.docx" />
        </FormControl>
        <Button leftIcon={<MdFileUpload />} onClick={handleSubmitFile} colorScheme="green" mt={4}>
          Upload File
        </Button>
      </Box> */}
    </Box>
  );
};

export default MaestraHome;