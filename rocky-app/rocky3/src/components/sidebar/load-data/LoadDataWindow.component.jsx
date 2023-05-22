import { useState, useRef } from 'react';
import PropTypes from 'prop-types';

import BaseDraggableWindow from '../../main-window/BaseDraggableWindow.component';
import appData from '../../../appdata';
import SampleDataButton from './SampleDataButton.component';
import TriangleTable from '../../data-components/TriangleTable.component';

// console.log(appData);

const SampleDataDropdown = ({
	isDataLoaded,
	setIsDataLoaded,
	result,
	setResult,
	setTriangleType,
	setWindowTitle,
}) => {
	const [sampleTriangle, setSampleTriangle] = useState(
		appData.sampleData[0].name,
	);
	setWindowTitle(appData.sampleData[0].name);

	let sampleData = appData.sampleData;
	setTriangleType('sample-data');

	const handleSelectChange = (event) => {
		setSampleTriangle(event.target.value);
	};

	return (
		<div className="sample-triangle-dropdown">
			{!isDataLoaded && (
				<>
					<p>Select sample triangle:</p>
					<select onSelect={handleSelectChange}>
						{sampleData.map((sample, index) => (
							<option key={index} value={sample.id}>
								{sample.name}
							</option>
						))}
					</select>

					<SampleDataButton
						// isDataLoaded={isDataLoaded}
						setIsDataLoaded={setIsDataLoaded}
						sampleTriangle={sampleTriangle}
						result={result}
						setResult={setResult}
					/>
				</>
			)}
		</div>
	);
};
SampleDataDropdown.propTypes = {
	isDataLoaded: PropTypes.bool,
	setIsDataLoaded: PropTypes.func,
	result: PropTypes.array,
	setResult: PropTypes.func,
	setTriangleType: PropTypes.func,
	setWindowTitle: PropTypes.func,
};

const LoadDataWindow = ({ title }) => {
	const [isDataLoaded, setIsDataLoaded] = useState(false);
	const [result, setResult] = useState([]);
	// const [isResult, setIsResult] = useState(false);
	const [triangleType, setTriangleType] = useState(null);
	const [windowTitle, setWindowTitle] = useState(title);
	const [triangleStyle, setTriangleStyle] = useState({
		// width: 150,
		// height: 300,
	});
	const triangleRef = useRef(null);

	return (
		<>
			<BaseDraggableWindow
				title={title}
				defautWidth={150}
				defaultHeight={300}
				windowType={'loadData'}
				startMinimized={false}
				triangleRef={triangleRef}
				triangleStyle={triangleStyle}
				setTriangleStyle={setTriangleStyle}
			>
				{(title === 'Sample Data' && (
					<div className="load-sample-data-window load-data-window">
						{!result && <h2>{windowTitle}</h2>}
						<SampleDataDropdown
							triangleStyle={triangleStyle}
							setTriangleStyle={setTriangleStyle}
							triangleRef={triangleRef}
							isDataLoaded={isDataLoaded}
							setIsDataLoaded={setIsDataLoaded}
							result={result}
							setResult={setResult}
							setWindowTitle={setWindowTitle}
							triangleType={triangleType}
							setTriangleType={setTriangleType}
						/>
						{result && (
							<TriangleTable
								data={result}
								triangleStyle={triangleStyle}
								setTriangleStyle={setTriangleStyle}
								triangleRef={triangleRef}
								setTriangleType={setTriangleType}
							/>
						)}
					</div>
				)) ||
					(title === 'Clipboard' && (
						<div className="load-clipboard-window load-data-window">
							<h2>Clipboard</h2>
							<p>
								Loading data from the clipboard has not been implemented yet.
							</p>
						</div>
					)) ||
					(title === 'Excel' && (
						<div className="load-excel-window load-data-window">
							<h2>Excel</h2>
							<p>
								Loading data from an excel file has not been implemented yet.
							</p>
						</div>
					)) ||
					(title === 'CSV' && (
						<div className="load-csv-window load-data-window">
							<h2>CSV</h2>
							<p>Loading data from a csv file has not been implemented yet.</p>
						</div>
					))}
			</BaseDraggableWindow>
		</>
	);
};
LoadDataWindow.propTypes = {
	title: PropTypes.string,
};

export default LoadDataWindow;
