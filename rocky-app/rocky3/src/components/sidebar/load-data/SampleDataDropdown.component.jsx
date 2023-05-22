/*
Sample Data Dropdown

This component is a dropdown menu that allows the user to select a sample
triangle from the appData file. The user can then click the "Get Started"
button to load the sample triangle into the app.

The list of sample triangles is generated from the appData file. The
sample triangle names are displayed in the dropdown menu.

*/

import PropTypes from 'prop-types';

import appData from '../../../appdata';

const SampleDataDropdown = ({ result, setTriangleType, setSampleTriangle }) => {
	let sampleData = appData.sampleData;
	setTriangleType('sample-data');

	const handleSelectChange = (event) => {
		setSampleTriangle(event.target.value);
	};

	return (
		<div className="sample-triangle-dropdown">
			{!result && (
				<>
					<p>Select sample triangle:</p>
					<select onSelect={handleSelectChange}>
						{sampleData.map((sample, index) => (
							<option key={index} value={sample.id}>
								{sample.name}
							</option>
						))}
					</select>
				</>
			)}
		</div>
	);
};
SampleDataDropdown.propTypes = {
	result: PropTypes.array,
	setTriangleType: PropTypes.func,
	setSampleTriangle: PropTypes.func,
};

export default SampleDataDropdown;
