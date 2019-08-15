import React from 'react';
import {NavBar, Icon, Carousel} from 'antd-mobile';
// import knowledgeBase from '../../assets/illness.json'

export default class Result extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            resultPic:'',
            height:0,
            results:[]
        }
    }
    componentWillMount(){
        if(this.props.location.query){
            this.setState({resultPic: this.props.location.query.resultpic,results: this.props.location.query.results,height:window.innerHeight});
        }else{
            this.setState({...this.props.location.state})
        }
    }
    componentDidMount() {
        if(this.props.location.query){
            let results = this.props.location.query.results;
            // results.push({else:true})
            console.log(results);
            this.props.history.replace('/Result', {...this.state,resultPic: this.props.location.query.resultpic,results: results,height:window.innerHeight});
        }
        
    }
    render() {
        return (
            <div>
                <NavBar
                    mode="light"
                    icon={<Icon type="left" />}
                    onLeftClick={() => this.props.history.push({pathname:'/Photo'})}>识别结果</NavBar>
                    <div className="resultPic" style={{backgroundImage:`url(${this.state.resultPic})`,height:this.state.height*0.35}}>
                        {/* <img src={require("../../assets/timg.jpg")} style={{width:"90%",height:"90%",margin:"10px auto"}}></img> */}
                    </div>
                    <div>
                        <Carousel
                            autoplay={false}
                            infinite
                            dots={true}
                            style={{height:(this.state.height-45)*0.55}}

                            // beforeChange={(from, to) => console.log(`slide from ${from} to ${to}`)}
                            // afterChange={index => console.log('slide to', index)}
                            >
                                {   
                                    this.state.results.map(val => {
                                        // if(!val.else){
                                            // if(val.detail && !val.detail.healthy){
                                            //     return <div className="clCard" key={val} style={{height:(this.state.height-45)*0.5}} >
                                            //             <div style={{height:"1rem"}}>   
                                            //                 <p>中草药名称：<span className="green">{val.name}</span></p>
                                            //                 <p>可能性：<span className="green">{val.score}</span></p>
                                            //             </div>
                                            //             <div className={`${val.imgName} plant leaf`} onClick={()=>this.detail(val.detail)}>
                                            //                 <p className="det">点击查看详情</p>
                                            //             </div>
                                            //         </div>
                                            // }else{
                                                return <div className="clCard" key={val} style={{height:(this.state.height-45)*0.5}}>
                                                        <div style={{height:"1rem"}}>   
                                                            <p>中草药名称：<span className="green">{val.name}</span></p>
                                                            <p>可能性：<span className="green">{val.score}</span></p>
                                                        </div>
                                                        <p className={`${val.class} plant leaf`}></p>
                                                    </div>
                                        //     }
                                            
                                        // }else{
                                            // return <div key={val} className="knowledge clCard" style={{height:(this.state.height-45)*0.5}} onClick={()=>{this.props.history.push({pathname:'/Baike'})}}>
                                            //             {/* <p>没有你想要的结果?</p> */}
                                            //             <div className="plant leaf" style={{backgroundImage:`url(${this.state.resultPic})`,marginTop:'1rem'}}>
                                            //                 <p className="noDet">以上结果都不对</p>
                                            //             </div>
                                            //             {/* <div className="btn">
                                            //                 <Button type="ghost"  icon={<img src={require('../../assets/icon_doc.png')} alt="" />}inline size="small">病虫害知识库</Button>                                                        </div> */}
                                            //         </div>
                                        // }
                                    })
                                }
                        </Carousel>
                    </div>
            </div>
            
        )
    }
}