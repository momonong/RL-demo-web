import { useState } from 'react'
import './App.css'
import Calculator from './components/Calculator'
import Photo from './components/Photo'
import { OperatorEnum } from './api'
import { Unstable_Grid2 as Grid, Container, Paper, Stack } from '@mui/material'

function App() {
  const [count, setCount] = useState(5)
  const [operator, setOperator] = useState<OperatorEnum>(OperatorEnum.MULTIPLY)


  return (
    <>
      <Container>
        <Stack 
          mb={2} 
          direction={'column'} 
          spacing={4} 
          border={1} 
          justifyContent={'center'} 
          alignItems={'center'} 
          height={'100px'}>
          <Paper>item1</Paper>
          <Paper>item2</Paper>
          <Paper>item3</Paper>
        </Stack>
        <Grid container border={1} spacing={2}>
          <Grid xs={4}>item {1}</Grid>
          <Grid xs={4}>item {1}</Grid>
        </Grid>
        <Calculator count={count} op={operator}/>
        <Photo />
      </Container>
    </>
  )
}

export default App
