import { useEffect, useState } from 'react';
import { Button, Grid, Typography, Box, Paper, Input, CircularProgress } from '@mui/material';
import { postComp2Field } from '../api';

function Comp2Field() {
    const [file, setFile] = useState<File>();
    const [originalSrc, setOriginalSrc] = useState<string>('');
    const [src, setSrc] = useState<string>('');
    const [loading, setLoading] = useState(false);

    const upload = async () => {
        if (!file) return;
        setLoading(true); // 開始加載
        const src = await postComp2Field(file);
        setLoading(false); // 停止加載
        setSrc(src);
    };

    useEffect(() => {
        if (file) {
            const reader = new FileReader();
            reader.addEventListener('load', () => {
                setOriginalSrc(reader.result as string);
            });
            reader.readAsDataURL(file);
        }
    }, [file]);

    return (
        <Box p={4}>
            <Paper elevation={3} style={{ padding: '20px' }}>
                <Grid container spacing={3}>
                    <Grid item xs={12} style={{ textAlign: 'center' }}>
                        <Input
                            type="file"
                            onChange={e => setFile((e.target as HTMLInputElement).files?.[0])}
                            fullWidth
                            style={{ display: 'none' }}
                            id="fileInput"
                        />
                        <label htmlFor="fileInput">
                            <Button variant="contained" color="primary" component="span">
                                Choose Image
                            </Button>
                        </label>
                        <Button 
                            variant="contained" 
                            color="primary" 
                            onClick={upload}
                            style={{ marginLeft: '20px' }}
                            disabled={loading} // 在加載時禁用按鈕
                        >
                            Process Image
                        </Button>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6" gutterBottom>
                            Original Image
                        </Typography>
                        {originalSrc && <img src={originalSrc} alt="Original" style={{ maxWidth: '100%' }} />}
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6" gutterBottom>
                            Processed Image
                        </Typography>
                        {loading && <CircularProgress style={{ marginLeft: '10px' }} />} {/* 加載指示器 */}
                        {src && <img src={src} alt="Processed" style={{ maxWidth: '100%' }} />}
                    </Grid>
                </Grid>
            </Paper>
        </Box>
    );
}

export default Comp2Field;
