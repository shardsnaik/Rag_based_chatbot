import React, { useState } from 'react';
import './medical_bot.css';
import {ToastContainer, toast, Bounce } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css';

const App = () => {

  const [activeMode, setActiveMode] = useState('analyzer');
  const [uploadedImage, setUploadedImage] = useState(null);

 // Separate chat states for each mode
  const [imageAnalyzerMessages, setImageAnalyzerMessages] = useState([]);
  const [chatAgentMessages, setChatAgentMessages] = useState([]);
  
  const [imageInputMessage, setImageInputMessage] = useState('');
  const [chatInputMessage, setChatInputMessage] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('med-gemma');

  const [chatHistory, setChatHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const unavailabe =()=>{
      toast.info('At this time service not available....', {
        position: "top-center",
        autoClose: 6000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "colored",
        transition: Bounce,
        } )
    }

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
        // Reset image analyzer chat when new image is uploaded
        setImageAnalyzerMessages([]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
        // Reset image analyzer chat when new image is uploaded
        setImageAnalyzerMessages([]);
      };
      reader.readAsDataURL(file);
    }
  };

    // API call for image analysis
  const callImageAnalysisAPI = async (message, imageFile) => {
    try {
      const formData = new FormData();
      formData.append('message', message);
      formData.append('image', imageFile);
      formData.append('model', selectedModel);

      const response = await fetch('/image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.response || data.message || 'Analysis complete.';
    } catch (error) {
      console.error('Image analysis API error:', error);
      return 'Sorry, there was an error analyzing your image. Please try again.';
    }
  };

  // API call for chat
  const callChatAPI = async (message) => {
    try {
      // const response = await fetch('http://127.0.0.1:8000/chat', {
      const response = await fetch('https://rag-based-chatbot-1.onrender.com/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          // model: selectedModel
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log(data)
      return data.response || data.message || 'I understand your question.';
    } catch (error) {
      console.error('Chat API error:', error);
      return 'Sorry, there was an error processing your message. Please try again.';
    }
  };

  const handleImageAnalyzerMessage = async () => {
    if (imageInputMessage.trim() && uploadedImage) {
      const newUserMessage = {
        id: Date.now(),
        text: imageInputMessage,
        sender: 'user',
        timestamp: new Date().toLocaleTimeString()
      };
      
      setImageAnalyzerMessages(prev => [...prev, newUserMessage]);
      setImageInputMessage('');
      setIsLoading(true);

      try {
        // Convert base64 image to file for API
        const response = await fetch(uploadedImage);
        const blob = await response.blob();
        const imageFile = new File([blob], 'medical-image.png', { type: 'image/png' });
        
        const botResponseText = await callImageAnalysisAPI(imageInputMessage, imageFile);
        
        const botMessage = {
          id: Date.now() + 1,
          text: botResponseText,
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString()
        };
        
        setImageAnalyzerMessages(prev => [...prev, botMessage]);
      } catch (error) {
        const errorMessage = {
          id: Date.now() + 1,
          text: 'Failed to analyze the image. Please check your connection and try again.',
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString()
        };
        setImageAnalyzerMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleChatAgentMessage = async () => {
    if (chatInputMessage.trim()) {
      const newUserMessage = {
        id: Date.now(),
        text: chatInputMessage,
        sender: 'user',
        timestamp: new Date().toLocaleTimeString()
      };
      
      setChatAgentMessages(prev => [...prev, newUserMessage]);
      setChatInputMessage('');
      setIsLoading(true);

      try {
        const botResponseText = await callChatAPI(chatInputMessage);
        
        const botMessage = {
          id: Date.now() + 1,
          text: botResponseText,
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString()
        };
        
        setChatAgentMessages(prev => [...prev, botMessage]);
      } catch (error) {
        const errorMessage = {
          id: Date.now() + 1,
          text: 'Failed to get response. Please check your connection and try again.',
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString()
        };
        setChatAgentMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

const handleKeyPress = (event, mode) => {
    if (event.key === 'Enter') {
      if (mode === 'image') {
        handleImageAnalyzerMessage();
      } else {
        handleChatAgentMessage();
      }
    }
  };

  // Fetch chat history from /memory endpoint
  const fetchChatHistory = async () => {
    try {      // http://127.0.0.1:8000/memory
      const response = await fetch('https://rag-based-chatbot-1.onrender.com/memory', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.text();
      console.log(data)
      
      // Parse the conversation history
      const conversations = parseConversationHistory(data);
      setChatHistory(conversations);
      
    } catch (error) {
      console.error('Failed to fetch chat history:', error);
      setChatHistory([]);
    }
  };

  // Parse conversation history from the text format
  const parseConversationHistory = (historyText) => {
     if (!historyText) return [];
    
    const conversations = [];
    const parts = historyText.split(/(?=Human\s*:)|(?=Assistant\s*:)/).filter(part => part.trim());
    
    parts.forEach((part, index) => {
      const trimmed = part.trim();
      if (trimmed.startsWith('Human')) {
        conversations.push({
          id: `human-${index}`,
          sender: 'user',
          text: trimmed.replace(/^Human\s*:\s*/, '').trim(),
          timestamp: new Date().toLocaleTimeString()
        });
      } else if (trimmed.startsWith('Assistant')) {
        conversations.push({
          id: `assistant-${index}`,
          sender: 'bot',
          text: trimmed.replace(/^Assistant\s*:\s*/, '').trim(),
          timestamp: new Date().toLocaleTimeString()
        });
      }
    });
    
    return conversations;
  };


  const handleModeSwitch = (mode) => {
    setActiveMode(mode);

    setIsLoading(false); // Reset loading state when switching modes
  // Fetch chat history when switching to any mode
    // if (mode) {
    //   fetchChatHistory();
    //   setShowHistory(true);
    // }
  };  

  const renderCenterPanel = () => {
    if (activeMode === 'analyzer') {
      if (uploadedImage) {
        return (
          <div className="chat-interface">
            <div className="uploaded-image">
              <img src={uploadedImage} alt="Uploaded medical scan" />
               <button 
                className="change-image-btn"
                onClick={() => setUploadedImage(null)}
              >Change Image</button> 

            </div>
             <div className="chat-messages">
              {imageAnalyzerMessages.length === 0 && (
                <div className="welcome-message">
                  <h3>Medical Image Analysis</h3>
                  <p>Ask questions about your uploaded medical image. I can help analyze X-rays, CT scans, MRIs, and other medical images.</p>
                </div>
              )}
              {imageAnalyzerMessages.map(message => (
                <div key={message.id} className={`message ${message.sender}`}>
                  <div className="message-content">
                    <span className="message-text">{message.text}</span>
                    <span className="message-time">{message.timestamp}</span>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message bot">
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="chat-input-container">
              <input
                type="text"
                value={imageInputMessage}
                onChange={(e) => setImageInputMessage(e.target.value)}
                onKeyPress={(e) => handleKeyPress(e, 'image')}
                placeholder="Ask about the medical image..."
                className="chat-input"
                 disabled={isLoading}
              />
             <button 
                onClick={handleImageAnalyzerMessage} 
                className="send-button"
                disabled={isLoading || !imageInputMessage.trim()}
              >
                {isLoading ? 'Analyzing...' : 'Send'}
              </button>
            </div>
          </div>
        );
      } else {
        return (
          <div className="upload-section" onDragOver={handleDragOver} onDrop={handleDrop}>
            <div className="upload-content">
              <div className="upload-icon">📁</div>
              <label htmlFor="file-upload" className="upload-label">
                Drag and Drop or Click to Upload Medical Image
              </label>
              <input 
                type="file" 
                id="file-upload" 
                className="upload-input" 
                onChange={handleFileUpload}
                accept="image/*"
              />
              <p className="upload-description">
                Supported formats: JPG, PNG, DICOM
              </p>
            </div>
          </div>
        );
      }
    } else {
      // Chat mode
      return (
        <div className="chat-interface">
          <div className="chat-messages">
            {chatAgentMessages.length === 0 && (
              <div className="welcome-message">
                <h3>Medical Chat Assistant</h3>
                <p>Ask me any medical questions and I'll provide evidence-based information. I can help with symptoms, treatments, medications, and general health advice.</p>
                <div className="sample-questions">
                  <h4>Sample questions:</h4>
                  <ul>
                    <li>"What are the symptoms of diabetes?"</li>
                    <li>"Explain the side effects of aspirin"</li>
                    <li>"What should I know about hypertension?"</li>
                  </ul>
                </div>
              </div>
            )}
            {chatAgentMessages.map(message => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div className="message-content">
                  <span className="message-text">{message.text}</span>
                  <span className="message-time">{message.timestamp}</span>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <div className="chat-input-container">
            <input
              type="text"
              value={chatInputMessage}
              onChange={(e) => setChatInputMessage(e.target.value)}
              onKeyPress={(e) => handleKeyPress(e, 'chat')}
              placeholder="Ask a medical question..."
              className="chat-input"
              disabled={isLoading}
            />
            <button onClick={handleChatAgentMessage} className="send-button"
             disabled={isLoading || !chatInputMessage.trim()}
            >
             {isLoading ? 'Thinking...' : 'Send'}
            </button>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="container">
{/* Chat History Panel (Left) */}
             <ToastContainer />

      {showHistory && (
        <div className="history-panel">
          <div className="history-header">
            <h3>Chat History</h3>
            <button 
              className="refresh-btn"
              onClick={fetchChatHistory}
              title="Refresh History"
            >
              🔄
            </button>
            <button 
              className="close-history-btn"
              onClick={() => setShowHistory(false)}
              title="Hide History"
            >
              ✕
            </button>
          </div>
          <div className="history-messages">
            {chatHistory.length === 0 ? (
              <div className="no-history">
                <p>No chat history available</p>
              </div>
            ) : (
              chatHistory.map((message, index) => (
                <div key={message.id || index} className={`history-message ${message.sender}`}>
                  <div className="history-message-content">
                    <div className="message-sender">
                      {message.sender === 'user' ? '👤 You' : '🤖 Assistant'}
                    </div>
                    <div className="message-text">{message.text}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Analysis Mode Section (Left) */}
      <div className="left-panel">
        <h2>Analysis Mode</h2>
        <button className={`mode-button ${activeMode === 'analyzer' ? 'active' : ' ' }`} onClick={ () => {handleModeSwitch('analyzer');   unavailabe()}} >
          📊 Medical Image Analyzer{imageAnalyzerMessages.length > 0 && <span className="message-count">{imageAnalyzerMessages.length}</span>}
        </button>
        <button 
          className={`mode-button ${activeMode === 'chat' ? 'active' : ' '}`}
          onClick={() => handleModeSwitch('chat')}
        >
         💬 Medical Chat Agent
          {chatAgentMessages.length > 0 && <span className="message-count">{chatAgentMessages.length}</span>}
        </button>
       {/* Show/Hide History Button */}
        <div className="history-controls">
          <button 
            className="history-toggle-btn"
            onClick={() => {
              if (!showHistory) {
                fetchChatHistory();
              }
              setShowHistory(!showHistory);
            }}
          >
            {showHistory ? '📋 Hide History' : '📋 Show History'}
          </button>
        </div>
      </div>


     {/* Dynamic Center Panel */}
      <div className="center-panel">
        {renderCenterPanel()}
      </div>
        

      {/* Select AI Model Section (Right) */}
      <div className="right-panel">
        <h2>Select AI Model</h2>
        <label className="model-option">
          <input type="radio" 
          name="ai-model" value="med-gemma"
          checked={selectedModel === 'med-gemma'}
          onChange={(e) => setSelectedModel(e.target.value)} />
          <span className="model-icon">🤖</span> Med-Gemma
          <p>Fine tunned Model for Medical Queries</p>
        </label>

        <label className="model-option">
          <input 
            type="radio" 
            name="ai-model" 
            value="claude-3"
            // checked={selectedModel === 'claude-3'}
            // onChange={(e) => setSelectedModel(e.target.value)}
          /><span className="model-icon">⭐</span> Claude 3
          <p>Comprehensive medical analysis</p>
        </label>

        <label className="model-option">
          <input 
          type="radio" name="ai-model" />
          <span className="model-icon">⚡</span> Gemini Pro Vision
          <p>Google's medical AI model</p>
        </label>

        <label className="model-option">
          <input type="radio" name="ai-model" />
          <span className="model-icon">🛡️</span> Med-PaLM 2
          <p>Specialized medical language model</p>
        </label>
      </div>
    </div>
  );
};

export default App;