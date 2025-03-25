import logo from './logo.svg';
import './App.css';
import React, { useState, useRef, useEffect } from 'react';
import { Container, TextField, Button, Typography } from '@mui/material';
import MessageWidget from './MessageWidget';
import AudioStreamer from './AudioStreamer';

const io = require('socket.io-client');
const SERVER_IP = '140.112.21.20';
const SERVER_PORT = '8080';
const socket = io(`http://${SERVER_IP}:${SERVER_PORT}`);

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);
  const [isConnected, setIsConnected] = useState(socket.connected);

  useEffect(() => {
    // Listen for connection events
    socket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to server');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from server');
    });

    socket.on('system_audio', (audio) => {
      // Handle incoming system audio
      handleSystemAudio(audio);
    });

    socket.on('user_text', (text) => {
      setMessages(messages => [...messages, { text, type: 'text' }]);
    });

    socket.on('system_text', (text) => {
      setMessages(messages => [...messages, { text, type: 'system_text' }]);
    });

    return () => {
      // Cleanup listeners on component unmount
      socket.off('connect');
      socket.off('system_audio');
      socket.off('user_text');
      socket.off('system_text');
    };
  }, []);

  const handleSystemAudio = (audio) => {
    // Process audio data - Placeholder
    console.log('Received system audio:', audio);
  };

  const sendMessage = () => {
    if (!inputText.trim()) return; // Prevent sending empty messages
    socket.emit('message', {
      type: 'text',
      data: inputText,
      input_timestamp: Date.now()
    });
    setMessages([...messages, { text: inputText, type: 'user_text' }]);
    setInputText('');
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      sendMessage();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom(); // Scroll to the bottom every time messages update
  }, [messages]);

  return (
    <div className="App">
      <header className="App-header">
        {/* <img src={logo} className="App-logo" alt="logo" /> */}
        <p>Hi~ Welcome to SpeechStreamingChatGPT!</p>
        <Typography variant="h6" style={{ color: isConnected ? 'green' : 'red' }}>
            {isConnected ? 'Connected' : 'Disconnected'}
        </Typography>
        <Container>
          <AudioStreamer socket={socket} />
          {messages.map((msg, index) => (
            <MessageWidget key={index} text={msg.text} type={msg.type} />
          ))}
          <div ref={messagesEndRef} />  {/* Invisible element for auto-scrolling */}
          <TextField
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            fullWidth
            placeholder="Type your message..."
          />
          <Button onClick={sendMessage} variant="contained">Send</Button>
        </Container>
      </header>
    </div>
  );
}

export default App;
