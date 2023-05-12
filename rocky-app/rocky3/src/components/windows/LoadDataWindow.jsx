import { useState } from 'react';
import PropTypes from 'prop-types';

import BaseDraggableWindow from './BaseDraggableWindow';
import appData from '../../appdata';
import SampleDataButton from '../buttons/SampleDataButton';
import TriangleTable from '../data-components/TriangleTable';

// console.log(appData);

const SampleDataDropdown = ({ isDataLoaded, setIsDataLoaded }) => {
	let sampleData = appData.sampleData;
	return (
		<div className="sample-triangle-dropdown">
			{!isDataLoaded && (
				<>
					<p>Select sample triangle:</p>
					<select>
						{sampleData.map((sample, index) => (
							<option key={index} value={sample.id}>
								{sample.name}
							</option>
						))}
					</select>

					<SampleDataButton setIsDataLoaded={setIsDataLoaded} />
				</>
			)}
			;
		</div>
	);
};
SampleDataDropdown.propTypes = {
	isDataLoaded: PropTypes.bool,
	setIsDataLoaded: PropTypes.func,
};

const LoadDataWindow = ({ title, triangleParentSize, triangleRef }) => {
	const [isDataLoaded, setIsDataLoaded] = useState(false);
	const [result, setResult] = useState(null);

	return (
		<>
			<BaseDraggableWindow
				title={title}
				defautWidth={150}
				defaultHeight={300}
				windowType={'loadData'}
				startMinimized={false}
			>
				{(title === 'Sample Data' && (
					<div className="load-sample-data-window load-data-window">
						<h2>Sample data</h2>
						<SampleDataDropdown
							triangleParentSize={triangleParentSize}
							triangleRef={triangleRef}
							isDataLoaded={isDataLoaded}
							setIsDataLoaded={setIsDataLoaded}
							result={result}
							setResult={setResult}
						/>
						{isDataLoaded && (
							<TriangleTable
								triangleParentSize={triangleParentSize}
								triangleRef={triangleRef}
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
	triangleParentSize: PropTypes.object,
	triangleRef: PropTypes.object,
};

export default LoadDataWindow;
