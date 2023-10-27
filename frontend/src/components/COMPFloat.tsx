import React, { useState, useEffect } from 'react';
import TextField from '@mui/material/TextField';

interface FloatInputProps {
  label?: string;
  onValueChange?: (value: number | null) => void;
}

function COMPFloat({ label = "Gamma", onValueChange = () => {} }: FloatInputProps) {
  const [value, setValue] = useState<string>('0.99');  // Set initial value to '0.99'
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    onValueChange(parseFloat(value));
  }, []);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = event.target.value;

    // Check if the input value is a valid float or empty string
    if (/^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/.test(inputValue) || inputValue === '') {
      setValue(inputValue);
      setError(false);
      onValueChange(inputValue ? parseFloat(inputValue) : null);
    } else {
      setError(true);
    }
  };

  return (
    <TextField
      label={label}
      variant="outlined"
      value={value}
      type="number"
      onChange={handleChange}
      error={error}
      InputProps={{
        style: { height: '60px', fontSize: '1.4rem', padding: '8px 8px' }
      }}
      helperText={error ? "Invalid float value" : ""}
    />
  );
}

export default COMPFloat;
