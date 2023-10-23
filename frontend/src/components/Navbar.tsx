import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { Link } from 'react-router-dom';

const pages = ['Smart RVE', 'DDPG', 'COMP', 'Comp2Field'];

function Navbar() {
  return (
    <AppBar position="fixed" style={{ width: '100%' }}>
        <Toolbar disableGutters>
            <img 
            src="https://static.wixstatic.com/media/37c673_8d76871adc8c49e181f4f9119139acb1~mv2.png/v1/fill/w_120,h_104,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/pic-10.png" 
            alt="Laboratory for Artificial Intelligence and Multiscale Modeling" 
            style={{ marginLeft: '20px', marginRight: '4px', height: '25px', width: 'auto' }} 
            />
            <Typography
            variant="h6"
            noWrap
            sx={{
                mr: 2,
                display: { xs: 'none', md: 'flex' },
                fontFamily: 'monospace',
                fontWeight: 700,
                letterSpacing: '.075rem',
                color: 'inherit',
                textDecoration: 'none',
            }}
            >
            LAiMM
            </Typography>
            <Typography
            variant="h5"
            noWrap
            sx={{
                mr: 2,
                display: { xs: 'flex', md: 'none' },
                flexGrow: 1,
                fontFamily: 'monospace',
                fontWeight: 700,
                letterSpacing: '.075rem',
                color: 'inherit',
                textDecoration: 'none',
            }}
            >
            LAiMM
            </Typography>
            {/* <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            {pages.map((page) => (
                <Button
                component={Link}
                to={`/${page.replace(' ', '-').toLowerCase()}`} // 轉換 "Smart RVE" 為 "/smart-rve"
                key={page}
                style={{ margin: '5px' }}
                sx={{ my: 2, color: 'white', display: 'block', textTransform: 'none'}}
                >
                {page}
                </Button>
            ))}
            </Box>
            <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
            {pages.map((page) => (
                <Button
                    component={Link}
                    to={`/${page.replace(' ', '-').toLowerCase()}`} // 轉換 "Smart RVE" 為 "/smart-rve"
                    key={page}
                    sx={{ my: 2, color: 'white', display: 'block', textTransform: 'none'}}
                >
                    {page}
                </Button>
                ))}
            </Box> */}
            <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
                {pages.map((page) => (
                <Button
                    key={page}
                    sx={{ my: 2, color: 'white', display: 'block' }}
                >
                    {page}
                </Button>
                ))}
            </Box>
        </Toolbar>
    </AppBar>
  );
}

export default Navbar;
