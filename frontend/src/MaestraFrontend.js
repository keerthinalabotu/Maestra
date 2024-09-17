import React, { useState } from 'react';
import MaestraHome from './components/MaestraHome';
import FileUpload from './components/FileUpload';
import IntroPage from './components/IntroPage';
// import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';

const MaestraFrontend = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [extractedResponse, setExtractedResponse] = useState('');

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return <IntroPage onStart={() => setCurrentStep(1)} />;
      case 1:
        return (
          <FileUpload
            onNext={(response) => {
              setExtractedResponse(response);  // Store the response from the upload
              setCurrentStep(2);  // Move to the next step
            }}
          />
          );
      // case 1:
      //   return <FileUpload onNext={() => setCurrentStep(2)} />;
      case 2:
        return <MaestraHome initialResponse={extractedResponse} />;
        // return <MaestraHome />;
      default:
        return <IntroPage onStart={() => setCurrentStep(1)} />;
    }
  };

  return (
    <div>
      {renderCurrentStep()}
    </div>
  );
};

export default MaestraFrontend;
