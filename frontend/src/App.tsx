import './App.css'
import Navbar from './components/Navbar'
import { OperatorEnum } from './api'
import { Unstable_Grid2 as Grid, Container, Paper, Stack } from '@mui/material'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { SmartRVE, DDPG, COMP, Comp2Field, Home } from './components/Home'


function App() {
  return (
    <>
      <Navbar />
      <Router>
        <div style={{ paddingTop: '64px' }}>
          <Navbar />
          <Routes>
            <Route path="/smart-rve" element={<SmartRVE />} />
            <Route path="/ddpg" element={<DDPG />} />
            <Route path="/comp" element={<COMP />} />
            <Route path="/comp2field" element={<Comp2Field />} />
            <Route path="/" element={<Home />} />
          </Routes>
        </div>
      </Router>
    </>
  );
}

export default App
