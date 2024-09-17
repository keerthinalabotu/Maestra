
import { Box, Button, Input, Text, useToast } from "@chakra-ui/react";
import React, { useState } from 'react';
// import { useNavigate } from 'react-router-dom';

const FileUpload = ({ onNext }) => {
  

  const [file, setFile] = useState(null);
  const [topic, setTopic] = useState("");
  const toast = useToast();
  // const navigate = useNavigate();

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleTopicChange = (event) => {
    setTopic(event.target.value);
  };

  const handleUpload = async () => {

    console.log('File:', file);
    console.log('Topic:', topic);
    
    if (!file || !topic) {
      toast({
        title: "Error",
        description: "Please select a file and enter a topic.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    console.log('WORKIGN>>>>>>');
    const formData = new FormData();
    formData.append('file', file);
    console.log('FORM DATA FILE:', file.name);
    console.log('File:', file);
    console.log('File size:', file.size, 'bytes');
    formData.append('topic', topic);

    // console.log(formdata);
    try {
      console.log('try');
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      console.log('fetched');

      const data = await response.json();
      console.log(data);
      if (!response.ok) throw new Error(data.message);

      toast({
        title: "Success",
        description: "File uploaded and processed successfully.",
        status: "success",
        duration: 5000,
        isClosable: true,
      });

      // setExtractedResponse(data.extracted_response);

      // navigate('http://localhost:8000/conversation_initial');

      // onUploadComplete(data.extracted_response);
      console.log(data);
      // onUploadComplete(data.extracted_response);
      console.log(data.extracted_response);

      onNext(data.extracted_response);
      console.log('WORKIGN>>>>>>');
    } catch (error) {
      console.log('fetched');
      console.error('Error:', error);
      toast({
        title: "Error",
        description: "Failed to upload file. Please try again.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Box
      bg="black"
      minHeight="100vh"
      display="flex"
      alignItems="center"
      justifyContent="center"
      p={4}
    >
      <Box
        bg="gray.800"
        p={6}
        borderRadius="md"
        boxShadow="lg"
        maxWidth="400px"
        width="100%"
        textAlign="center"
      >
        <Text fontSize="2xl" color="white" mb={6}>
          What do you want to learn today?
        </Text>
        <Input type="file" color="white" mb={4} onChange={handleFileChange} />
        <Input
          placeholder="Enter a topic"
          variant="filled"
          bg="white"
          mb={4}
          width="100%"
          value={topic}
          sx={{
            '::placeholder': {
              color: 'gray.600', // Set placeholder color if needed
            },
            color: 'white', // Set input text color
          }}
          onChange={handleTopicChange}

        />
        <Button
          colorScheme="blue"
          size="lg"
          onClick={handleUpload}
          _hover={{ bg: "blue.600", color: "white" }}
        >
          Begin learning with Maestra!
        </Button>
      </Box>
    </Box>
  );
};

export default FileUpload;
