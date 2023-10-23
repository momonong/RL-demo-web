import { useState } from "react";
import Grid from '@mui/material/Grid';
import CNNGridInput from "./CNNGrid";
import CNNNumberInput from "./CNNNumber";
import { clearPlot, postSmartRVE } from "../api";  // 請確保這是正確的導入路徑
import Button from '@mui/material/Button';

export default function SmartRVE() {
    const [src, setSrc] = useState<string>('');
    const [selectedCells, setSelectedCells] = useState<number[]>([]);
    const [otherParameters, setOtherParameters] = useState<any>({});

    const updateOtherParameters = (params: any) => {
        setOtherParameters(params);
    }

    const fetchData = async () => {
        console.log("Other Parameters:", otherParameters);  // 打印 otherParameters 的值
        const newSrc = await postSmartRVE(selectedCells, otherParameters);
        setSrc(newSrc);
    }

    const handleClearPlot = async () => {
        try {
            await clearPlot();
            setSrc('');  // 清除圖片
            console.log("Plot cleared successfully");
        } catch (error) {
            console.error("Failed to clear plot:", error);
        }
    }
    
    return (
        <Grid container spacing={3} marginTop={'100px'}>
            <Grid item xs={12} md={3}>
                <CNNGridInput onSelectedCellsChange={setSelectedCells} />
            </Grid>
            <Grid item xs={12} md={9}>
                <CNNNumberInput onParametersChange={updateOtherParameters} />
            </Grid>
            <Grid item xs={12}>
                <Button variant="contained" color="primary" onClick={fetchData} style={{ marginTop: '16px', marginRight: '16px' }}>
                    Get Processed Image
                </Button>
                <Button variant="contained" color="primary" onClick={handleClearPlot} style={{ marginTop: '16px' }}>
                    Clear Plot
                </Button>
                {/* 如果有圖片URL，則顯示圖片 */}
                {src && <img src={src} alt="Processed" style={{ maxWidth: '100%', display: 'block', margin: '0 auto' }} />}
            </Grid>
        </Grid>
    );    
}

