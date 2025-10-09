import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Second_page from './compo/second_page/center/Center';
import Llm_pagw from './compo/llm_page/Hheader';
import Med_bot from './compo/medical_bot/Medical_bot'
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<App />} />
        <Route path='/llm' element={<Llm_pagw />} />
        <Route path='/second_page' element={<Second_page />} />
        <Route path='/med_page' element={<Med_bot />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

// If you want to start measuring performance in your