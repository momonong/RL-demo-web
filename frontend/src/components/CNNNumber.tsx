import React, { useState, useEffect } from 'react';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Unstable_Grid2';

function CNNNumberInput({ onParametersChange }: Props) {
    const [values, setValues] = useState<ValuesType>({
        angular_of_weaving: 90,
        width_of_yarn: 0.9,
        height_of_yarn: 0.3,
        space: 1.8,
        epoxy_E: 20000,
        epoxy_v: 0.4,
        epoxy_yield_strength_1: 3,
        epoxy_plastic_strain_1: 0,
        epoxy_yield_strength_2: 600,
        epoxy_plastic_strain_2: 0.3,
        fibre_density: 2550,
        fibre_linear_density: 0.00056,
        fibre_E1: 72000,
        fibre_E2: 72000,
        fibre_E3: 72000,
        fibre_G12: 30000,
        fibre_G23: 30000,
        fibre_G13: 30000,
        fibre_v1: 0.2,
        fibre_v2: 0.2,
        fibre_v3: 0.2,
        selected_cells: []
    });

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        const updatedValues = {
            ...values,
            [name]: value
        };
        setValues(updatedValues);
        onParametersChange(updatedValues);
    };

    // 使用useEffect在組件初始化時傳遞預設值
    useEffect(() => {
        onParametersChange(values);
    }, []);  // 空依賴數組確保此useEffect只在組件掛載時運行    

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Grid container spacing={2} sx={{ maxWidth: '90%' }}>
        <Grid  xs={12} md={3}>
            <TextField style={{ marginBottom: '16px' }} label="Angular of Weaving" type="number" name="angular_of_weaving" value={values.angular_of_weaving} onChange={handleChange}  
                InputProps={{endAdornment: <InputAdornment position="end">degree(°)</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Width of Yarn" type="number" name="width_of_yarn" value={values.width_of_yarn} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">mm</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Height of Yarn" type="number" name="height_of_yarn" value={values.height_of_yarn} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">mm</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Space" type="number" name="space" value={values.space} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">mm</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Epoxy E" type="number" name="epoxy_E" value={values.epoxy_E} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Epoxy v" type="number" name="epoxy_v" value={values.epoxy_v} onChange={handleChange} />
        </Grid>
        <Grid xs={12} md={3}>
            <TextField style={{ marginBottom: '16px' }} label="Epoxy Yield Strength 1" type="number" name="epoxy_yield_strength_1" value={values.epoxy_yield_strength_1} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Epoxy Plastic Strain 1" type="number" name="epoxy_plastic_strain_1" value={values.epoxy_plastic_strain_1} onChange={handleChange} />
            <TextField style={{ marginBottom: '16px' }} label="Epoxy Yield Strength 2" type="number" name="epoxy_yield_strength_2" value={values.epoxy_yield_strength_2} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Epoxy Plastic Strain 2" type="number" name="epoxy_plastic_strain_2" value={values.epoxy_plastic_strain_2} onChange={handleChange} />
            <TextField style={{ marginBottom: '16px' }} label="Fibre Density" type="number" name="fibre_density" value={values.fibre_density} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">kg/m<sup>3</sup></InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Fibre Linear Density" type="number" name="fibre_linear_density" value={values.fibre_linear_density} onChange={handleChange}
                InputProps={{endAdornment: <InputAdornment position="end">kg/m</InputAdornment>}}/>
        </Grid>
        <Grid xs={12} md={3}>
            <TextField style={{ marginBottom: '16px' }} label="Fibre E1" type="number" name="fibre_E1" value={values.fibre_E1} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Fibre E2" type="number" name="fibre_E2" value={values.fibre_E2} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Fibre E3" type="number" name="fibre_E3" value={values.fibre_E3} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Fibre G12" type="number" name="fibre_G12" value={values.fibre_G12} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Fibre G23" type="number" name="fibre_G23" value={values.fibre_G23} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
            <TextField style={{ marginBottom: '16px' }} label="Fibre G13" type="number" name="fibre_G13" value={values.fibre_G13} onChange={handleChange} 
                InputProps={{endAdornment: <InputAdornment position="end">MPa</InputAdornment>}}/>
        </Grid>
        <Grid xs={12} md={3}>
            <TextField style={{ marginBottom: '16px' }} label="Fibre v1" type="number" name="fibre_v1" value={values.fibre_v1} onChange={handleChange} />
            <TextField style={{ marginBottom: '16px' }} label="Fibre v2" type="number" name="fibre_v2" value={values.fibre_v2} onChange={handleChange} />
            <TextField style={{ marginBottom: '16px' }} label="Fibre v3" type="number" name="fibre_v3" value={values.fibre_v3} onChange={handleChange} />
            {/* 這裡不包括 selected_cells，因為它是由網格選擇功能處理的 */}
        </Grid>
      </Grid>
    </Box>
  );
}

export default CNNNumberInput;
