// src/App.js
import React, { useState, useRef, useEffect } from 'react';
import './header.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    
    const userMessage = { text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/llm_bot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputValue })
      });
      
      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      const botMessage = { 
        text: data.answer, 
        sender: 'bot',
        cached: data.cached
      };
      console.log(data)
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { 
        text: `Error: ${error.message}`, 
        sender: 'bot',
        isError: true 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-placeholder" />
          <div className="header-text">
            <h1>Enterprise AI Assistant</h1>
            <p>Powered by DeepSeek-R1</p>
          </div>
        </div>
      </header>

      {/* Chat Container */}
      <div className="chat-container">
        <div className="messages-wrapper">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-logo" />
              <h2>Enterprise AI Assistant</h2>
              <p className="welcome-subtext">
                Ask questions about your enterprise data and get AI-powered insights
              </p>
              <div className="example-queries">
                <h3>Example Queries</h3>
                <ul>
                  <li>Summarize last quarter's sales report</li>
                  <li>What are our top performing products?</li>
                  <li>Generate customer segmentation analysis</li>
                </ul>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div 
                key={index} 
                className={`message-container ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
              >
                <div 
                  className={`message-bubble ${msg.sender === 'user' ? 'user-bubble' : 'bot-bubble'} ${
                    msg.isError ? 'error-bubble' : ''
                  }`}
                >
                  <div className="message-sender">
                    {msg.sender === 'user' ? 'You' : 'Enterprise Assistant'}
                  </div>
                  <p className="message-text">{msg.text}</p>
                  {msg.sender === 'bot' && !msg.isError && (
                    <div className="message-meta">
                      {msg.cached 
                        ? 'Served from cache' 
                        : 'Generated in real-time'}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message-container bot-message">
              <div className="message-bubble bot-bubble">
                <div className="message-sender">Enterprise Assistant</div>
                <div className="loading-indicator">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="messages-end" />
        </div>
      </div>

      {/* Input Area */}
      <footer className="input-footer">
        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-wrapper">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about your enterprise data..."
              className="message-input"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className={`send-button ${(!inputValue.trim() || isLoading) ? 'disabled-button' : ''}`}
            >
              Send
            </button>
          </div>
          <p className="footer-note">
            Enterprise AI Assistant v1.0 · All queries are securely processed
          </p>
        </form>
      </footer>
    </div>
  );
}

export default App;