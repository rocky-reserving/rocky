import { useState } from 'react';
import PropTypes from 'prop-types';
import TriangleTable from '../data-components/TriangleTable';
import appData from '../../appdata';

const SampleDataButton = ({
	triangleRef,
	setIsDataLoaded,
	result,
	setResult,
}) => {
	const [loading, setLoading] = useState(false);

	function handleClick() {
		setLoading(true);

		fetch(appData.api.load_taylor_ashe, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ user_id: 'your_user_id' }),
		})
			.then((response) => {
				if (response.ok) {
					return response.json();
				} else {
					throw new Error('Error fetching data: ' + response.statusText);
				}
			})
			.then((data) => {
				console.log('Data:', data);
				// Parse the JSON string and convert it into an array of objects
				const parsedData = Object.entries(JSON.parse(data.result)).map(
					([key, value]) => ({ ...value, id: key }),
				);
				setResult(parsedData);
				setIsDataLoaded(true);
			})
			.catch((error) => {
				console.error('Error:', error);
				setIsDataLoaded(false);
			})
			.finally(() => {
				setLoading(false);
			});
	}

	return (
		<div>
			<button onClick={handleClick} disabled={loading}>
				{loading ? 'Loading...' : 'Load Taylor Ashe'}
			</button>
			{result && (
				<div>
					<h3>Result:</h3>
					<TriangleTable data={result} ref={triangleRef} />
				</div>
			)}
		</div>
	);
};
SampleDataButton.propTypes = {
	triangleRef: PropTypes.object,
	setIsDataLoaded: PropTypes.func,
	result: PropTypes.array,
	setResult: PropTypes.func,
};

export default SampleDataButton;
