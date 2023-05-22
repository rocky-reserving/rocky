import { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

import BaseDraggableWindow from '../BaseDraggableWindow.component';
import SampleDataDropdown from './SampleDataDropdown.component';
import SampleDataButton from './SampleDataButton.component';
import TriangleTable from '../../data-components/TriangleTable.component';

import appData from '../../../appdata';

const LoadDataWindow = ({ title }) => {
	const [isDataLoaded, setIsDataLoaded] = useState(false);
	const [result, setResult] = useState([]);
	const [triangleType, setTriangleType] = useState(null);
	const [windowTitle, setWindowTitle] = useState(title);
	const triangleRef = useRef(null);

	const defaultTriangle = appData.sampleData[0].name;
	const [sampleTriangle, setSampleTriangle] = useState(defaultTriangle);

	// when the sample triangle changes, update the window title to match
	useEffect(() => {
		setWindowTitle(windowTitle);
	}, [sampleTriangle]);

	useEffect(() => {
		if (result.length !== 0) {
			setIsDataLoaded(true);
		} else {
			setIsDataLoaded(false);
		}
	}, [result]);

	return (
		<>
			<BaseDraggableWindow
				title={title}
				defautWidth={150}
				defaultHeight={300}
				windowType={'loadData'}
				startMinimized={false}
				triangleRef={triangleRef}
				// triangleStyle={triangleStyle}
				// setTriangleStyle={setTriangleStyle}
			>
				{(title === 'Sample Data' && (
					<div className="load-sample-data-window load-data-window">
						{/* Sample Data heading */}
						{!isDataLoaded && <h2>{windowTitle}</h2>}

						{/* Sample data dropdown */}
						{!isDataLoaded && (
							<SampleDataDropdown
								result={result}
								setSampleTriangle={setSampleTriangle}
								triangleType={triangleType}
								setTriangleType={setTriangleType}
							/>
						)}

						{/* Sample data button */}
						{!isDataLoaded && (
							<SampleDataButton
								sampleTriangle={sampleTriangle}
								setResult={setResult}
							/>
						)}

						{/* Triangle table */}
						{isDataLoaded && <TriangleTable data={result} />}
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
