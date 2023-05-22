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

const SampleDataDropdown = ({ setTriangleType, setSampleTriangle }) => {
	let sampleData = appData.sampleData;

	const handleSelectChange = (event) => {
		setSampleTriangle(event.target.value);
		setTriangleType('sample-data');
	};

	return (
		<div className="sample-triangle-dropdown">
			<select onChange={handleSelectChange}>
				{sampleData.map((sample, index) => (
					<option key={index} value={sample.name}>
						{sample.name}
					</option>
				))}
			</select>
		</div>
	);
};
SampleDataDropdown.propTypes = {
	triangleType: PropTypes.string,
	setTriangleType: PropTypes.func,
	setSampleTriangle: PropTypes.func,
};

export default SampleDataDropdown;
