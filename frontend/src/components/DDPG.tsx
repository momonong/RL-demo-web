import React, { useState } from 'react';
import { Grid, TextField, Button, CircularProgress } from '@mui/material';
import { postDDPG } from '../api'; 
// @ts-ignore
import ReactGifPlayer from 'react-gif-player';

const DDPG: React.FC = () => {
  const [inputValue, setInputValue] = useState<string>(''); // I changed the name to avoid confusion
  const [gifSrc, setGifSrc] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);


  const handleFetchGif = async () => {
    const parsedFloatValue = parseFloat(inputValue);
    setLoading(true);  // 啟動加載

    try {
        if (isNaN(parsedFloatValue) || parsedFloatValue < 0.3 || parsedFloatValue > 0.7) {
            throw new Error('請輸入有效的數字，範圍在 0.3 到 0.7 之間。');  
        }

        setGifSrc('');  // 清空現有的GIF URL

        const gifObjectURL = await postDDPG(parsedFloatValue);

        setTimeout(() => {
            setGifSrc(gifObjectURL);  // 延遲設定新的GIF URL
        }, 50);

    } catch (error) {
        if (error instanceof Error && error.message === '請輸入有效的數字，範圍在 0.3 到 0.7 之間。') {
            alert(error.message);
        } else {
            console.error('錯誤:', error);
            alert('無法獲取 GIF 圖片。');
        }
    } finally {
        setLoading(false);  // 無論成功還是失敗，都停止加載
    }
  };

const handleReplayGif = () => {
    if (gifSrc) {
        const newImage = new Image();

        newImage.onload = () => {
            // 確保清除先前的GIF
            // @ts-ignore
            setGifSrc(null);

            // 使用setTimeout來等待下一個事件循環
            setTimeout(() => {
                setGifSrc(gifSrc);
            }, 0);
        };

        // 啟動圖像加載
        newImage.src = gifSrc;
    }
};



return (
  <Grid container spacing={1} marginTop={'20px'}>
    
    {/* 左邊的TextField */}
    <Grid item xs={12} md={6}>
    <TextField
          fullWidth
          label="Target Porosity ( 0.3 ~ 0.7 )"
          variant="outlined"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          type="number"
          InputProps={{
            style: { height: '56px', fontSize: '1.3rem', padding: '8px 8px' }
          }}
          style={{ maxWidth: '300px' }}  // 縮小TextField的最大寬度
        />
    </Grid>

    {/* 右邊的Button */}
    <Grid item xs={12} md={6}>
        <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center' }}>
            <Button 
            variant="contained" 
            color="primary" 
            onClick={handleFetchGif} 
            style={{ height: '56px', fontSize: '1rem', marginLeft: '2px' }}
            >
            Fetch GIF
            </Button>
            
            <Button 
            variant="contained" 
            color="primary" 
            onClick={handleReplayGif} 
            style={{ height: '56px', fontSize: '1rem', marginLeft: '6px' }}  // 讓兩個按鈕的高度和字體大小相同
            >
            Replay GIF
            </Button>
        </div>
    </Grid>


    {/* 加載指示器 */}
    {loading && <Grid item xs={12}><CircularProgress style={{ margin: '16px auto', display: 'block' }} /></Grid>}

    {/* 顯示GIF圖片 */}
    <Grid item xs={12} marginTop={4}>
                {gifSrc && (
                    <>
                        {gifSrc && <ReactGifPlayer gif={gifSrc} autoplay={true} style={{ maxWidth: '80%', display: 'block', margin: '0 auto' }} />}

                    </>
                )}
    </Grid>
  </Grid>
);
}

export default DDPG;
