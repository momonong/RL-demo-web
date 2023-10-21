import { useState } from "react"
import { calculate } from "../api"

const Calculator = ({count, op}: {count: number, op: string}) => {
    const [operator, setOperator] = useState<string>(op)
    const [a, setA] = useState<number>(count)
    const [b, setB] = useState<number>(count)
    const [result, setResult] = useState<number>(0)
    const onClick = async () => {
        const result = await calculate(operator, a, b)
        setResult(result)
    }
    return <div>
            Calculator 
            <input type="number" value={a} onChange={e => setA(Number(e.target.value))} />
            <select name="operator" id="operator" onChange={e => setOperator(e.target.value)}>
                {[{name: 'add', text: '+'},
                 {name: 'sub', text: '-'},
                 {name: 'mul', text: '*'},
                 {name: 'div', text: '/'}
                 ].map(({name, text}) => <option key={name} value={name} selected={operator === name}>{text}</option>)}
                {/* <option value="add">+</option>
                <option value="sub">-</option>
                <option value="mul">*</option>
                <option value="div">/</option> */}
            </select>
            <input type="number" value={b} onChange={e => setB(Number(e.target.value))} />
            <button onClick={onClick}> Calculate {a} {operator} {b} </button>
            <div> Result: {result} </div>
        </div>
}

export default Calculator