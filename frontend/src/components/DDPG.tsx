import React, { useState } from 'react';

const DDPGIceCrystalModel: React.FC = () => {
  const [inputValue, setInputValue] = useState<number | string>('');
  const [gifUrl, setGifUrl] = useState<string | null>(null);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const floatValue = parseFloat(inputValue as string);
    if (isNaN(floatValue) || floatValue < 0.3 || floatValue > 0.7) {
      alert('請輸入有效的數字，範圍在 0.3 到 0.7 之間。');
      return;
    }

    try {
      const response = await fetch(`/model_ddpg_ice_crystal?request_ratio=${floatValue}`);
      if (!response.ok) {
        throw new Error('獲取 GIF 圖片失敗');
      }
      const gifBlob = await response.blob();
      const gifObjectURL = URL.createObjectURL(gifBlob);
      setGifUrl(gifObjectURL);
    } catch (error) {
      console.error('錯誤:', error);
      alert('無法獲取 GIF 圖片。');
    }
  };

  return (
    <div>
      <h1>DDPG Ice Crystal 模型</h1>
      <form onSubmit={handleSubmit}>
        <label>
          請輸入浮點數 (0.3 - 0.7)：
          <input
            type="number"
            step="0.01"
            min="0.3"
            max="0.7"
            value={inputValue}
            onChange={handleInputChange}
            required
          />
        </label>
        <button type="submit">提交</button>
      </form>
      {gifUrl && (
        <div>
          <h2>生成的 GIF 圖片：</h2>
          <img src={gifUrl} alt="Generated GIF" />
        </div>
      )}
    </div>
  );
};

export default DDPGIceCrystalModel;
