import { useState, useRef, useEffect } from 'react';
// import { Triangle } from './classes/Triangle.js';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';
// import BaseDraggableWindow from './components/windows/BaseDraggableWindow';
import Sidebar from './components/sidebar/Sidebar.component';
import MainWorkspace from './components/main-window/MainWorkspace.component';

// let t = new Triangle();

function App() {
	const [isFreshlyLoaded, setIsFreshlyLoaded] = useState(true);
	const [loadDataWindows, setLoadDataWindows] = useState({});
	// const [modelSelectionWindows, setModelSelectionWindows] = useState({});
	// const [modelValidationWindows, setModelValidationWindows] = useState({});
	// const [visualizationWindows, setVisualizationWindows] = useState({});
	const [triangleParentSize, setTriangleParentSize] = useState({
		// width: 'auto',
		// height: 'auto',
	});

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

	useEffect(() => {
		if (triangleRef.current) {
			const resizeObserver = new ResizeObserver((entries) => {
				const { width, height } = entries[0].contentRect;
				setTriangleParentSize({ width, height });
			});

			resizeObserver.observe(triangleRef.current);

			return () => {
				resizeObserver.disconnect();
			};
		}
	}, [triangleRef]);

	// Set the parent component's size based on the state
	// const parentStyle = {
	// 	width: parentSize.width,
	// 	height: parentSize.height,
	// 	// ...
	// };

	return (
		<>
			<div id="app">
				<a href="#" target="_blank">
					<img src={viteLogo} className="logo" alt="Vite logo" />
				</a>
				<a href="#" target="_blank">
					<img src={reactLogo} className="logo react" alt="React logo" />
				</a>
			</div>
			<h1 className="main-title">rocky</h1>
			<div className="card">
				{/* <button onClick={() => setCount((count) => count + 1)}>
					count is {count}
				</button> */}
				<div id="sidebar">
					<Sidebar
						isSidebarExpanded={isSidebarExpanded}
						setIsSidebarExpanded={setIsSidebarExpanded}
						onClickNew={onClickNew}
						onAddLoadDataWindow={onAddLoadDataWindow}
					/>
				</div>

				<MainWorkspace
					loadDataWindows={loadDataWindows}
					triangleParentSize={triangleParentSize}
					triangleRef={triangleRef}
					isFreshlyLoaded={isFreshlyLoaded}
					onClickLoadButton={onClickLoadButton}
				/>
			</div>
		</>
	);
}

export default App;
