import React from 'react';
import { Button} from 'antd-mobile';



export default class Home extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            
        }
    }
    render() {
        return (
            <div>
                <Button type="primary" inline onClick={()=>{this.props.history.push({pathname:'/Photo'})}}>智能识别</Button>
                {/* <Button type="primary" inline onClick={()=>{this.props.history.push({pathname:'/Baike'})}}>病虫害百科</Button> */}
            </div>
        )
    }
}