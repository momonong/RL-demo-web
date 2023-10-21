import { useState } from 'react'
import './App.css'
import Calculator from './components/Calculator'
import Photo from './components/Photo'

function App() {
  const [count, setCount] = useState(5)
  const [operator, setOperator] = useState('mul')


  return (
    <>
      <Calculator count={count} op={operator}/>
      <Photo />
    </>
  )
}

export default App
