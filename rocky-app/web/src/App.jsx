import { useState, useRef } from 'react';

// import { initializeApp } from 'firebase/app';
// import { getFirestore } from 'firebase/firestore';

import './App.css';
import Navbar from './components/navbar/Navbar.component';
import Landing from './components/landing/Landing.component';
import Sidebar from './components/sidebar/Sidebar.component';
import MainWorkspace from './components/main-window/MainWorkspace.component';
import Footer from './components/footer/Footer.component';
// import LandingButton from './components/LandingButton.component';

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
// const firebaseConfig = {
// 	apiKey: 'AIzaSyB-nbq5C6-Nlbi6SzYlcsU15ZWntHlRB5Y',
// 	authDomain: 'rocky-dev-1.firebaseapp.com',
// 	projectId: 'rocky-dev-1',
// 	storageBucket: 'rocky-dev-1.appspot.com',
// 	messagingSenderId: '580547406823',
// 	appId: '1:580547406823:web:713c737b33d77f51bc6b42',
// 	measurementId: 'G-ER1N4ERGW0',
// };

// Initialize Firebase
// const firebaseApp = initializeApp(firebaseConfig);
// const db = getFirestore(firebaseApp);

const App = () => {
	const [isFreshlyLoaded, setIsFreshlyLoaded] = useState(true);
	const [loadDataWindows, setLoadDataWindows] = useState({});
	// const [modelSelectionWindows, setModelSelectionWindows] = useState({});
	// const [modelValidationWindows, setModelValidationWindows] = useState({});
	// const [visualizationWindows, setVisualizationWindows] = useState({});
	// const [triangleParentSize, setTriangleParentSize] = useState({
	// width: 'auto',
	// height: 'auto',
	// });

	const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);
	const triangleRef = useRef(null);

	const onClickNew = () => {
		setLoadDataWindows({});
		setIsFreshlyLoaded(true);
	};

	const onAddLoadDataWindow = (title) => {
		const id = Math.random().toString(36).substr(2, 9);
		setLoadDataWindows({
			...loadDataWindows,
			[id]: {
				title: title,
				defautWidth: 150,
				defaultHeight: 300,
				windowType: 'loadData',
				startMinimized: false,
			},
		});
		setIsFreshlyLoaded(false);
	};

	const onClickLoadButton = () => {
		setIsFreshlyLoaded(false);
		setIsSidebarExpanded(true);
	};

	return (
		<div
			className={`app-main ${
				isSidebarExpanded ? 'sidebar-expanded' : 'sidebar-collapsed'
			}`}
		>
			<Navbar />
			<Landing
				isFreshlyLoaded={isFreshlyLoaded}
				onClickLoadButton={onClickLoadButton}
				isSidebarExpanded={isSidebarExpanded}
			/>
			{/* <div id="app" className="landing-header"> */}
			{/* <div className="landing-above-button"> */}
			{/* <div className="logo-container"></div> */}

			{/* {isFreshlyLoaded && (
						<LandingButton onClickLoadButton={onClickLoadButton} />
					)} */}
			{/* </div> */}
			{/* <div className="landing-image-section"></div> */}
			{/* </div> */}

			<Sidebar
				isSidebarExpanded={isSidebarExpanded}
				setIsSidebarExpanded={setIsSidebarExpanded}
				onClickNew={onClickNew}
				onAddLoadDataWindow={onAddLoadDataWindow}
			/>

			<MainWorkspace
				loadDataWindows={loadDataWindows}
				// triangleParentSize={triangleParentSize}
				triangleRef={triangleRef}
				isFreshlyLoaded={isFreshlyLoaded}
				onClickLoadButton={onClickLoadButton}
			/>

			<Footer />
		</div>
	);
};

export default App;
