import React from 'react';
import Cropper from 'react-cropper';
import 'cropperjs/dist/cropper.css';
import {Button,Toast} from 'antd-mobile';

export default class CropperPic extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            imgSrc:null,
            height:0,
            file:null
        }
    }
    componentDidMount() {
        if(this.props.location.query){
            const fileReader = new FileReader()
            fileReader.onload = (e) => {
            const dataURL = e.target.result
            this.setState({imgSrc: dataURL,height:window.innerHeight-47,file:this.props.location.query})
            }
            fileReader.readAsDataURL(this.props.location.query.file)
        }else{
            this.props.history.push({pathname:'/'})
        }
        
    }
    async submit(){
        let that = this;
        let result = this.refs.cropper.getCroppedCanvas({width: 299,height: 299});
        let resultpic = result.toDataURL();
        result.toBlob(async function(blob) {
            let formData = new FormData();
            // let test = new Date().getTime()+".png";
            // debugger
            // formData.append("file",blob,test);
            formData.append("file",that.state.file.file);
            
            Toast.loading('识别中', 50000, () => {
                console.log('Load complete !!!');
              });
            //上传
            let resp = await fetch("/app/upload",{
                method:"POST",
                headers:{},
                body:formData
            })
            let responseJson = await resp.json();
            // let responseJson = {"code": 0, "result": [{"label_id": "7", "name": "茶树炭疽病", "score": "75.18%"}, {"label_id": "4", "name": "茶叶叶枯病",
            // "score": "20.29%"}, {"label_id": "8", "name": "茶苗白绢病", "score": "4.13%"}]}
            Toast.hide();
            that.props.history.push({pathname:'/Result',query:{resultpic:resultpic,results:responseJson.result}})
          });
        
    }
    render() {
        return (
            <div>
                <div style={{height:'100%'}}>
                    <Cropper
                        src={this.state.imgSrc}
                        ref="cropper"
                        // src={require("../../assets/timg.jpg")}
                        style={{height:this.state.height}}
                        // className="company-logo-cropper"
                        // ref={cropper => this.cropper = cropper}
                        viewMode={1} 
                        cropBoxMovable={false}
                        cropBoxResizable={false}
                        dragMode={'move'}
                        aspectRatio={1/1}
                    />
                    <div className="clBtn">
                        <Button inline onClick={()=>{this.props.history.push({pathname:'/Photo'})}}>取消</Button>
                        <Button inline onClick={()=>this.submit()}>确定</Button>
                    </div>
                    
                </div>
                
            </div>
        )
    }
}