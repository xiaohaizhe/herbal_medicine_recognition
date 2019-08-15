// import React from 'react';
// import {BrowserRouter as Router, Route, Switch,withRouter} from 'react-router-dom';

import Photo from './Photo/Photo';
// import Baike from './Baike/Baike';
import CropperPic from './Cropper/Cropper';
import Result from './Result/Result'
// import Detail from './Detail/Detail'

// const getRouter = () => (
//     <Router>
//         <div className="App">
//             {/* <ul>
//                 <li><Link to="/">首页</Link></li>
//                 <li><Link to="/Photo">Page1</Link></li>
//             </ul> */}
//             <Switch>
//                 <Route exact path="/" component={Photo}/>
//                 <Route path="/Photo" component={Photo}/>
//                 <Route path="/Baike" component={Baike}/>
//                 <Route path="/Cropper" component={CropperPic}/>
//                 <Route path="/Result" component={Result}/>
//                 <Route path="/Detail" component={Detail}/>
//             </Switch>
//         </div>
//     </Router>
// );

// export default getRouter;


const routes = [
    {
        path: '/',
        component: Photo,
        exact: true,
    },
    // {
    //     path: '/Baike',
    //     component: Baike,
    //     children: [
    //         {
    //             path: '/Baike/Detail',
    //             component: Detail
    //         }
    //     ]
    // },
    {
        path: '/Cropper',
        component: CropperPic
    },
    {
        path: '/Result',
        component: Result
    },{
        path: '/Photo',
        component: Photo
    },
];
export default routes