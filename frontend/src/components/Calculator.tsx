import { useEffect, useState } from "react"
import { calculate } from "../api"
import { OperatorEnum } from "../api"
import { TextField, MenuItem, Select } from "@mui/material"

const Calculator = ({count, op}: {count: number, op: OperatorEnum}) => {
    const [operator, setOperator] = useState<OperatorEnum>(op)
    const [a, setA] = useState<number>(count)
    const [b, setB] = useState<number>(count)
    const [result, setResult] = useState<number>(0)
    const update = async () => {
        const result = await calculate(operator, a, b)
        setResult(result)
    }

    useEffect(() => {
        if (a && b && operator) update()
    }, [a, b, operator])

    return <div>
            Calculator 
            <TextField type="number" variant="filled" color='secondary' value={a} onChange={e => setA(Number(e.target.value))} />
            <Select name="operator" id="operator" value={operator} onChange={e => setOperator(e.target.value as OperatorEnum)}>
                {[{name: OperatorEnum.ADD, text: '+'},
                 {name: OperatorEnum.DIVIDE, text: '-'},
                 {name: OperatorEnum.MULTIPLY, text: '*'},
                 {name: OperatorEnum.DIVIDE, text: '/'}
                 ].map(({name, text}) => <MenuItem key={name} value={name}>{text}</MenuItem>)}
                {/* <option value="add">+</option>
                <option value="sub">-</option>
                <option value="mul">*</option>
                <option value="div">/</option> */}
            </Select>
            <TextField type="number" variant="filled" color='secondary' value={b} onChange={e => setB(Number(e.target.value))} />
            <div> Result: {result} </div>
        </div>
}

export default Calculator