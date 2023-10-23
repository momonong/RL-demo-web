export enum OperatorEnum {
    ADD = 'add',
    SUBTRACT = 'sub',
    MULTIPLY = 'mul',
    DIVIDE = 'div',
}

export const calculate  = async (operator: string, a: number, b: number ) => {
    // const params = new URLSearchParams({operator: 'sub', a: '5', b: '4'})
    // const response = await fetch(`http://localhost:8000/math?${params.toString()}`)
    const response = await fetch(`http://localhost:8000/math`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({operator, a, b}),
    })

    const {result} = await response.json()
    return result
}

export const postAiArtPortrait = async (file: string | Blob) => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch(`http://127.0.0.1:8000/ai-art-portrait`, {
        method: 'POST',
        body: formData,
    })
    const blob = await response.blob()
    return URL.createObjectURL(blob)
    console.log(response)
}

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

// api.ts
export const clearPlot = async () => {
  const response = await fetch(`http://127.0.0.1:8000/clear_plot`, {
      method: 'POST',
  });

  if (!response.ok) {
      throw new Error(`API call failed with status ${response.status}`);
  }

  return response.json();
}


