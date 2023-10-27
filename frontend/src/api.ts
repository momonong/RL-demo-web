// CNN Plastic
// Smart RVE
export const postSmartRVE = async (selectedCells: number[], otherParameters: any) => {
  // 創建一個物件來存儲所有參數
  const requestBody = {
    ...otherParameters,
    selected_cells: selectedCells
  };

  const response = await fetch(`http://127.0.0.1:8000/model_smart_rve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
      throw new Error(`API call failed with status ${response.status}`);
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);  // 返回圖片的URL
}

export const clearPlot = async () => {
  const response = await fetch(`http://127.0.0.1:8000/clear_plot`, {
      method: 'POST',
  });

  if (!response.ok) {
      throw new Error(`API call failed with status ${response.status}`);
  }

  return response.json();
}

// DDPG
// Ice Crystal 
export const postDDPG = async (requestRatio: number): Promise<string> => {
  // 組裝請求體
  const requestBody = {
      request_ratio: requestRatio
  };

  const response = await fetch(`http://127.0.0.1:8000/model_ddpg_ice_crystal`, {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
          'Accept': 'image/gif'
      },
      body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
      throw new Error(`API call failed with status ${response.status}`);
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);  // 返回GIF圖片的URL
}

// COMP
// Composites design
export const postCOMP = async (requestRatio: number, gridInput: number[]): Promise<string> => {
  // 組裝請求體
  const requestBody = {
    gamma: requestRatio,
    selected_cells: gridInput
};
  const response = await fetch(`http://127.0.0.1:8000/model_comp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
      throw new Error(`API call failed with status ${response.status}`);
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);  // 返回圖片的URL
}

// HRRL
// Comp2Field
export const postComp2Field = async (file: string | Blob) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`http://127.0.0.1:8000/model_comp2field`, {
      method: 'POST',
      body: formData,
  })
  const blob = await response.blob()
  return URL.createObjectURL(blob)
}