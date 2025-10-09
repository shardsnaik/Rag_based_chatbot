import React, { useEffect, useState } from "react";
import "./home.css";
import robo from '../../images/33.png'
import robo2 from '../../images/22.png'
import { Link } from "react-router-dom";
import {ToastContainer, toast, Bounce } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css';


const ChatbotPage = () => {
const [firtimg, secimg ] =useState([robo])

useEffect(() =>{
  const img_robo = [robo, robo2]
  let indx = 0
  const imginterval = setInterval(()=>{
    indx = (indx + 1 ) % img_robo.length
    secimg(img_robo[indx])
  }, 10000) 
  return () => clearInterval(imginterval)
},[])

const tostload =()=>{
  toast.info('Currently Not Available', {
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
  return (
    <div className="chatbot-container">
      <ToastContainer/>
      <div className="chatbot-header">
        <h1>Available AI Services</h1>
      </div>
      <div className="chatbot-content">
        <div className="text-section">
          <h2>CHATBOT</h2>
          {/* <h3>technology</h3> */}
          <p>
          Unlock Personalized Insights with Our Cutting-Edge RAG Model.
          </p>
          <Link to={'/second_page'} >
          <button className="get-started-button">RAG Model</button></Link>
          <p>
          Experience the Future with Our Advanced LLM Model Capable of Generating Human-like Text.
          </p>
          <Link to={'/llm'} >
          <button onClick={tostload} className="get-started-button">LLM Model</button></Link>
          <p>
          Our Smart Medical doctor is here.
          </p>
          <Link to={'/med_page'} >
          <button className="get-started-button">Medical Bot</button></Link>
          <p>
          Experience the medical report analysis, medical bot agent.
          </p>
        </div>
        <div className="image-section">
          <img
            src={firtimg} // Replace with your image path
            alt="Chatbot"
            className="chatbot-image"
          />
        </div>
      </div>
    </div>
  );
};

export default ChatbotPage;
