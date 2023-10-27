import { useState } from 'react';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import COMPGrid from './COMPGrid';
import COMPFloat from './COMPFloat';
import { postCOMP } from '../api';  

function COMP() {
  const [selectedCells, setSelectedCells] = useState<number[]>([]);
  const [floatValue, setFloatValue] = useState<number | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);  // 加載狀態

  const handleSelectedCellsChange = (cells: number[]) => {
    setSelectedCells(cells);
  };

  const handleFloatValueChange = (value: number | null) => {
    setFloatValue(value);
  };

  const handleSubmit = async () => {
    setIsLoading(true);  // 開始加載
    try {
        const imageUrl = await postCOMP(floatValue as number, selectedCells);
        setSrc(imageUrl);
    } catch (error) {
        console.error("Error sending data to the API:", error);
    } finally {
        setIsLoading(false);  // 結束加載
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', marginTop: '10px' }}>
      {/* 輸入介面 */}
      <div style={{ height: '500px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <COMPGrid onSelectedCellsChange={handleSelectedCellsChange} />
        <COMPFloat label="Gamma" onValueChange={handleFloatValueChange} />
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleSubmit}
          size="large" 
          style={{ width: '100%' }}  
          disabled={isLoading}  // 當正在加載時禁用按鈕
        >
          {isLoading ? <CircularProgress size={24} /> : 'Submit'}  
        </Button>
      </div>
  
      {/* 圖片或佔位方塊 */}
      <div style={{ height: '500px', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: 'white' }}>
        {src && <img src={src} alt="Processed" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />}
      </div>
    </div>
  );
}

export default COMP;
