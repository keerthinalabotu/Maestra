import { Box, Button, Text } from "@chakra-ui/react";

const IntroPage = ({ onStart }) => {
  return (
    <Box
      bg="black"
      height="100vh"
      display="flex"
      alignItems="center"
      justifyContent="center"
      flexDirection="column"
    >
      <Text fontSize="6xl" color="white" fontWeight="bold" mb={6}>
        Maestra
      </Text>
      <Button
        colorScheme="blue"
        size="lg"
        onClick={onStart}
        _hover={{ bg: "blue.600", color: "white" }}
      >
        Start Learning!
      </Button>
    </Box>
  );
};

export default IntroPage;