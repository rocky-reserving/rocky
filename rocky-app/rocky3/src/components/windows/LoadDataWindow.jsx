import BaseDraggableWindow from './BaseDraggableWindow';
import PropTypes from 'prop-types';
import appData from '../../appdata';
import SampleDataButton from '../buttons/SampleDataButton';

// console.log(appData);

const SampleDataDropdown = () => {
	let sampleData = appData.sampleData;
	return (
		<div className="sample-triangle-dropdown">
			<p>Select sample triangle:</p>
			<select>
				{sampleData.map((sample, index) => (
					<option key={index} value={sample.id}>
						{sample.name}
					</option>
				))}
				{/* <option value="taylor-ashe">Taylor-Ashe Paid Loss</option>
				<option value="dahms-rpt">Dahms Reported Loss</option>
				<option value="dahms-paid">Dahms Paid Loss</option> */}
			</select>
			<SampleDataButton />
		</div>
	);
};

const LoadDataWindow = ({
	title,
	// defautWidth,
	// defaultHeight,
	// windowType,
	// startMinimized = false,
}) => {
	// // function to get the items for the load data window
	// const getLoadDataItems = () => {
	// 	appData.sidebarItems.forEach((item) => {
	// 		if (item.id === 'load-data') {
	// 			return item.items;
	// 		}
	// 	});
	// };
	// const load = getLoadDataItems();

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
						<SampleDataDropdown />
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
	// defautWidth: PropTypes.number,
	// defaultHeight: PropTypes.number,
	// windowType: PropTypes.string,
	// startMinimized: PropTypes.bool,
};

export default LoadDataWindow;
