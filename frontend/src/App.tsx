import './App.css'
import Navbar from './components/Navbar'
import SmartRVE from './components/SmartRVE';
import DDPG from './components/DDPG';
import COMP from './components/COMP';
import Comp2Field from './components/Comp2Field';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // 使用 BrowserRouter

function App() {
  return (
    <Router> {/* 使用 BrowserRouter */}
      <Navbar />
      <Routes>
        <Route path="/smart-rve" element={<SmartRVE />} />
        <Route path="/ddpg" element={<DDPG />} />
        <Route path="/comp" element={<COMP />} />
        <Route path="/comp2field" element={<Comp2Field />} />
        <Route path="/" element={<SmartRVE />} /> {/* 默认路由 */}
      </Routes>
    </Router>
  );
}

export default App;
