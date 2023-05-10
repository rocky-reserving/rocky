// import { useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';
import DraggableWindow from './components/DraggableWindow';
import Sidebar from './components/Sidebar';

function App() {
	// const [count, setCount] = useState(0);

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
			<h1>rocky</h1>
			<div className="card">
				{/* <button onClick={() => setCount((count) => count + 1)}>
					count is {count}
				</button> */}
				<div id="sidebar">
					<Sidebar />
				</div>
				<div id="main-workspace">
					{/* <DraggableWindow
						title="Default"
						defautWidth={400}
						defaultHeight={300}
						windowType={'default'}
						startMinimized={true}
					></DraggableWindow> */}
					<div id="load-data">
						<section>
							<h2>Load Data</h2>
							<DraggableWindow
								title="Load Data"
								defautWidth={400}
								defaultHeight={300}
								windowType={'loadData'}
								startMinimized={false}
							></DraggableWindow>
						</section>
					</div>
					<section id="model-selection">
						<DraggableWindow
							title="Model Selection"
							defautWidth={400}
							defaultHeight={300}
							windowType={'modelSelection'}
							startMinimized={true}
						></DraggableWindow>
					</section>
					<section id="model-validation">
						<DraggableWindow
							title="Model Validation"
							defautWidth={400}
							defaultHeight={300}
							windowType={'modelValidation'}
							startMinimized={true}
						></DraggableWindow>
					</section>
				</div>
			</div>
		</>
	);
}

export default App;
