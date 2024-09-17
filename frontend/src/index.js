// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import './index.css';
// import App from './App';

// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <App/> 
//     </App>
// );

// index.js

import React from 'react';
import ReactDOM from 'react-dom';
import './index.css'; // Import global styles
import { ChakraProvider } from '@chakra-ui/react';
import App from './App'; // Import the root component
// import reportWebVitals from './reportWebVitals'; // Optional: for measuring performance

// Render the App component into the DOM
ReactDOM.render(
  <React.StrictMode>
    <ChakraProvider>
        <App />
    </ChakraProvider>
  </React.StrictMode>,
  document.getElementById('root') // This is where the React app gets attached in the HTML
); 

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
reportWebVitals();
