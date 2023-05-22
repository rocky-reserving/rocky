import { useState } from 'react';
import PropTypes from 'prop-types';
import appData from '../../../appdata';

const SampleDataButton = ({ setResult, sampleTriangle }) => {
	const [loading, setLoading] = useState(false);

	const handleClick = () => {
		console.log('Current sampleTriangle: ', sampleTriangle);
		setLoading(true);

		console.log('sampleTriangle:', sampleTriangle);
		let apiURL = appData.api[sampleTriangle];
		console.log('apiURL:', apiURL);

		fetch(apiURL, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ user_id: 'your_user_id' }),
		})
			.then((response) => {
				console.log('response:', response);
				if (response.ok) {
					return response.json();
				} else {
					// setSampleTriangle('');
					throw new Error('Error fetching data: ' + response.statusText);
				}
			})
			.then((data) => {
				// console.log('Data:', data);
				// Parse the JSON string and convert it into an array of objects
				const parsedData = Object.entries(JSON.parse(data.result)).map(
					([key, value]) => ({ ...value, id: key }),
				);
				// console.log('Parsed Data:', parsedData);
				setResult(parsedData);
				// setSampleTriangle(sampleTriangle);
			})
			.catch((error) => {
				console.error('Error:', error);
				// setSampleTriangle('');
			})
			.finally(() => {
				setLoading(false);
			});
	};

	return (
		<div>
			<button onClick={handleClick} disabled={loading}>
				{loading ? 'Loading...' : 'Load Taylor Ashe'}
			</button>
		</div>
	);
};
SampleDataButton.propTypes = {
	// triangleRef: PropTypes.object,
	// setIsDataLoaded: PropTypes.func,
	// result: PropTypes.array,
	setResult: PropTypes.func,
	sampleTriangle: PropTypes.string,
	setSampleTriangle: PropTypes.func,
};

export default SampleDataButton;
