import './App.css'
import Navbar from './components/Navbar'
// import { OperatorEnum } from './api'
// import { Unstable_Grid2 as Grid, Container, Paper, Stack } from '@mui/material'
// import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
// import { DDPG, COMP, Comp2Field, Home } from './components/Home'
import SmartRVE from './components/SmartRVE';
import  DDPG from './components/DDPG';


function App() {
  return (
    <>
      <Navbar />
      {/* <Router>
          <Navbar />
          <Routes>
            <Route path="/smart-rve" element={<SmartRVE />} />
            <Route path="/ddpg" element={<DDPG />} />
            <Route path="/comp" element={<COMP />} />
            <Route path="/comp2field" element={<Comp2Field />} />
            <Route path="/" element={<Home />} />
      </Routes>
      </Router>*/} 
      <SmartRVE />
      {/* <DDPG /> */}
    </>
  );
}

export default App
