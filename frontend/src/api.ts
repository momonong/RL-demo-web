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