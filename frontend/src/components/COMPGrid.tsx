import { useState } from 'react';
import { styled } from '@mui/material/styles';
import Grid from '@mui/material/Unstable_Grid2';
import Box from '@mui/material/Box';
import ButtonBase from '@mui/material/ButtonBase';

// COMPGridInput Component
interface COMPGridInputProps {
  onSelectedCellsChange?: (selectedCells: number[]) => void;
}

const CustomButton = styled(ButtonBase)(({ 'data-mirrored': mirrored, selected }: { 'data-mirrored': boolean, selected: boolean }) => ({
  width: '100%',
  height: '100%',
  backgroundColor: selected ? (mirrored ? '#a0a0a0' : 'black') : mirrored ? '#f0f0f0' : 'lightgrey',
  transition: 'background-color 0.3s',
}));

function COMPGrid({ onSelectedCellsChange = () => {} }: COMPGridInputProps) {
  const [selected, setSelected] = useState<number[]>([]);

  const handleClick = (index: number) => {
    if (index % 4 > 1) return; // Only allow selection for the left half

    let updatedSelected: number[] = [];
    
    if (selected.includes(index)) {
        updatedSelected = selected.filter((item) => item !== index);
    } else {
        updatedSelected = [...selected, index];
    }
  
    updatedSelected.sort((a, b) => a - b);
  
    setSelected(updatedSelected);
    onSelectedCellsChange(updatedSelected);
  };

  const isMirroredSelected = (index: number) => {
    const mirroredIndex = index - (index % 4) + (3 - (index % 4));
    return selected.includes(mirroredIndex);
  };
  
  return (
    <Box sx={{ width: '100%', height: 'auto' }}>
        <Grid container spacing={1}>
            {[...Array(16)].map((_, index) => (
            <Grid xs={3} md={3} key={index}>
                <div style={{ paddingTop: '100%', position: 'relative' }}>
                    <CustomButton
                    style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
                    selected={index % 4 <= 1 ? selected.includes(index) : isMirroredSelected(index)}
                    data-mirrored={index % 4 > 1}
                    onClick={() => handleClick(index)}
                    />
                </div>
            </Grid>
            ))}
        </Grid>
    </Box>
  );
}

export default COMPGrid;