import { useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';
// import BaseDraggableWindow from './components/windows/BaseDraggableWindow';
import Sidebar from './components/Sidebar';
import MainWorkspace from './components/MainWorkspace';

function App() {
	const [isFreshlyLoaded, setIsFreshlyLoaded] = useState(true);
	const [loadDataWindows, setLoadDataWindows] = useState({});
	// const [modelSelectionWindows, setModelSelectionWindows] = useState({});
	// const [modelValidationWindows, setModelValidationWindows] = useState({});
	// const [visualizationWindows, setVisualizationWindows] = useState({});
	const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);

	function onClickNew() {
		setLoadDataWindows({});
		setIsFreshlyLoaded(true);
	}

	function onAddLoadDataWindow(title) {
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
	}

	function onClickLoadButton() {
		setIsFreshlyLoaded(false);
		setIsSidebarExpanded(true);
	}

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
					isFreshlyLoaded={isFreshlyLoaded}
					onClickLoadButton={onClickLoadButton}
				/>
			</div>
		</>
	);
}

export default App;
