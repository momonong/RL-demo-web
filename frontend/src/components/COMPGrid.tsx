import { useState } from 'react';
import { styled } from '@mui/material/styles';
import Grid from '@mui/material/Unstable_Grid2';
import Box from '@mui/material/Box';
import ButtonBase from '@mui/material/ButtonBase';

// COMPGridInput Component
interface COMPGridInputProps {
  onSelectedCellsChange?: (selectedCells: number[]) => void;
}

const CustomButton = styled(ButtonBase)(({ 'data-mirrored': mirrored, selected }: { theme?: any, 'data-mirrored': boolean, selected: boolean }) => ({
  width: '100%',
  height: '100%',
  backgroundColor: selected ? (mirrored ? '#a0a0a0' : 'black') : mirrored ? '#f0f0f0' : 'lightgrey',
  transition: 'background-color 0.3s',
}));

function COMPGrid({ onSelectedCellsChange = () => {} }: COMPGridInputProps) {
  const [selected, setSelected] = useState<number[]>([]);

  const handleClick = (index: number) => {
    // 确定点击只能在左半边
    if (index % 8 >= 4) return;

    let updatedSelected: number[] = [];
    if (selected.includes(index)) {
      updatedSelected = selected.filter((item) => item !== index);
    } else {
      updatedSelected = [...selected, index];
    }

    // 排序选择的索引
    updatedSelected.sort((a, b) => a - b);

    // 转换为4x8网格的索引
    const convertedSelected = updatedSelected.map(i => {
      // 计算4x8的行和列
      const row = Math.floor(i / 8);
      const col = i % 8;
      return row * 4 + col;
    });

    setSelected(updatedSelected);
    onSelectedCellsChange(convertedSelected);
};

  // This function determines if the cell on the right should be mirrored
  const isMirroredSelected = (index: number) => {
    const row = Math.floor(index / 8);
    const column = index % 8;
    // Calculate mirrored index for the left side of the grid
    const mirroredIndex = row * 8 + (7 - column);
    return selected.includes(mirroredIndex);
  };
  
  return (
    <Box sx={{ width: '100%', height: 'auto' }}>
        <Grid container spacing={1}>
            {[...Array(64)].map((_, index) => (
            <Grid xs={1.5} md={1.5} key={index}>
                <div style={{ paddingTop: '100%', position: 'relative' }}>
                    <CustomButton
                    style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
                    selected={selected.includes(index)}
                    data-mirrored={index % 8 >= 4}
                    onClick={() => handleClick(index)}
                    sx={{ 
                      backgroundColor: index % 8 < 4 ? (selected.includes(index) ? 'black' : 'lightgrey') 
                                                      : (isMirroredSelected(index) ? '#a0a0a0' : '#f0f0f0')
                    }}
                    />
                </div>
            </Grid>
            ))}
        </Grid>
    </Box>
  );
}

export default COMPGrid;