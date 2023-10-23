import { styled } from '@mui/material/styles';
import Grid from '@mui/material/Unstable_Grid2';
import Box from '@mui/material/Box';
import ButtonBase from '@mui/material/ButtonBase';
import { useState } from 'react';

interface CNNGridInputProps {
  onSelectedCellsChange: (selectedCells: number[]) => void;
}

const CustomButton = styled(ButtonBase)(({ selected }: { selected: boolean }) => ({
  width: '100%',
  height: '100%',
  backgroundColor: selected ? 'black' : 'lightgrey',
  transition: 'background-color 0.3s',
}));

function CNNGridInput({ onSelectedCellsChange }: CNNGridInputProps) {
  const [selected, setSelected] = useState<number[]>([]);

  const handleClick = (index: number) => {
    let updatedSelected: number[] = [];
    
    if (selected.includes(index)) {
        updatedSelected = selected.filter((item) => item !== index);
    } else {
        updatedSelected = [...selected, index];
    }
  
    updatedSelected.sort((a, b) => a - b);
  
    setSelected(updatedSelected);
    onSelectedCellsChange(updatedSelected);  // 調用回調函數
};
  
  return (
    <Box sx={{ width: '100%', height: 'auto' }}>
        <Grid container spacing={1}>
            {[...Array(25)].map((_, index) => (
            <Grid xs={2.4} md={2.4} key={index}>
                <div style={{ paddingTop: '100%', position: 'relative' }}>
                    <CustomButton
                    style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
                    selected={selected.includes(index)}
                    onClick={() => handleClick(index)}
                    />
                </div>
            </Grid>
            ))}
        </Grid>
    </Box>
  );
}

export default CNNGridInput;
