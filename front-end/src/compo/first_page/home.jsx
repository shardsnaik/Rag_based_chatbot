import React from "react";
import "./Chatbot.css";

const ChatbotPage = () => {
  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h1>Company Logo</h1>
      </div>
      <div className="chatbot-content">
        <div className="text-section">
          <h2>CHATBOT</h2>
          <h3>technology</h3>
          <p>
            Lorem ipsum dolor sit amet, consectetur elit, sed do eiusmod tempor
            incididunt ut dolore magna aliqua. Ut enim ad minim nostrud
            consequat.
          </p>
          <button className="get-started-button">Get Started</button>
        </div>
        <div className="image-section">
          <img
            src="/path/to/chatbot-image.png" // Replace with your image path
            alt="Chatbot"
            className="chatbot-image"
          />
        </div>
      </div>
    </div>
  );
};

export default ChatbotPage;
