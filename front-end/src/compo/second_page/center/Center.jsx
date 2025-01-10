import React, { useState} from "react";
import "./center.css";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlay } from '@fortawesome/free-solid-svg-icons'
import {jsPDF} from 'jspdf'
import {ToastContainer, toast, Bounce } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css';

const Center = () => {
  const [save_con_pdf, set_save_con_pdf] = useState([])
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [pdfname, setpdfname] = useState('No file uploaded')
    const [data_available, setdata_available] = useState(false)
  
  
    
  const tostload =()=>{
    toast.info('PDF Uploading....', {
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
  const tostmes =()=>{
    toast.success('PDF Uploaded Successfully', {
      position: "top-center",
      autoClose: 3000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      progress: undefined,
      theme: "colored",
      transition: Bounce,
      } )
  }

  const handleDownloadPDFs = () => {
    const doc = new jsPDF();
    doc.setFontSize(13);
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 10; // Set margins
    const maxWidth = pageWidth - margin * 2; // Calculate maximum width for text
    let y = 10; // Starting Y position


    save_con_pdf.forEach((item) => {
      const text = `${item.sender} : ${item.text}`;
      const lines = doc.splitTextToSize(text, maxWidth); 
    lines.forEach((line) => {
      doc.text(line, margin, y);
      y += 5; //  line height
      if (y > doc.internal.pageSize.getHeight() - 10) {
        // If the text goes beyond the page, add a new page
        doc.addPage();
        y = 10; 
      }
    });
  });
    doc.save('Chat.pdf');
  }
  
    const handleInputChange = (e) => {
      setInput(e.target.value);
    };
  
    const handleSendMessage = async () => {
      if (input.trim() === '') return;
  
      const newMessage = { text: input, sender: 'user' };
      setMessages([...messages, newMessage]);

      set_save_con_pdf([...save_con_pdf, newMessage])
      
      try {
        const response = await fetch('https://rag-llm-chatbot.onrender.com/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query: input }), // Ensure the payload matches the expected format
        });
  
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
  
        const data = await response.json();
        const botMessage = { text: data.answer, sender: 'bot' };

        setMessages((prevMessages) => [...prevMessages, botMessage]);

        set_save_con_pdf((prev) =>[
          ...prev, botMessage
        ])

        console.log(data);
        // console.log('exferfer');
        // console.log(data.answer);
        // console.log(data.query);
      } catch (error) {
        console.error('Error communicating with the API:', error);
      }
  
      setInput('');
    };
  
    const handleFileUpload = async (e) => {
      const file = e.target.files[0];
      console.log(file);
      if (file){
        setpdfname(file.name);
        tostload()
  
        const formData = new FormData();
        formData.append('file', file);
  
        try {
          const res = await fetch('https://rag-llm-chatbot.onrender.com/upload/', {
            method: 'POST',
            body: formData,
          });
          console.log(res);
          
          if (!res.ok){
            throw new Error('Network response was not ok');
          }
  
          const data = await res.json();
          console.log('pdf uploaded', data);
          setdata_available(true)
          tostmes()
          
        }
        catch (error){
          console.error('Error communicating with the API:', error);  
      }}
    }
    const handleKeyDown = (e) => {
      if (e.key === 'Enter') {
        handleSendMessage();
      }}
  


return(
  <>
    <div className="nav-chat-header">
      <div className="nav_top_bot-info">
        <div className="nav_top_bot-avatar">B</div>
        <div className="nav_top_bot-details">
          <h3 className="nav_top_bot-name">RAG-based Chatbot</h3>
          <p className="nav_top_bot-description">Chat with this bot..! that use external data sources to answer questions accurately and contextually.
          </p>
        </div>
      </div>
      <div className="nav_top_chat-actions">
    
        {/* <span className="dt">Download Chat</span> */}
        <button onClick={handleDownloadPDFs} className="nav_top_action-button">⬇</button>
        <button className="nav_top_action-button">☰</button>
        <button className="nav_top_action-button">✖</button>
      </div>
    </div>
 <div className='centr-con' >
 <ToastContainer />
<div className="main-container">
 <div className="upload-section">
   <div className="pdf-name">{pdfname}</div>
   <div className="upload-button">

   <input type="file" accept='.pdf, .txt, .json, .csv' onChange={handleFileUpload} style={{opacity:0, position: 'absolute'}} id='file-upload' />
    <button  className='upl-btn' >Upload</button>
   </div>
 </div>

 <div className="chat-section">
   <div className={!data_available ? 'temp-header' : 'chat-header'}>RAG Model Integrated with Chat-Gpt and {pdfname.length > 10 ? `${pdfname.substring(0, 10)}...` : pdfname}</div>

   <div className="chat-messages">
     {messages.map((message, index) => (
       <div key={index} className={`message-${message.sender}`}>
         {message.text}
       </div>
     ))}
   </div>
  </div>
  
 </div>
 

</div>
<div id="ms-in-parent">
<div className="message-input">
     <input
       type="text"
       placeholder="Type your message..."
       value={input}
       onChange={handleInputChange}
       onKeyDown={handleKeyDown}
     />
     <div className='send-icon' onClick={handleSendMessage} >
     <FontAwesomeIcon icon={faPlay} size="2xl" />
     </div>
   </div>
</div>


</>
)}
export default Center;


