import { useState } from 'react';

const SampleDataButton = () => {
	const [result, setResult] = useState(null);
	const [loading, setLoading] = useState(false);

	function handleClick() {
		setLoading(true);

		fetch('http://localhost:1234/rockyapi/load-taylor-ashe', {
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
				setResult(data.result);
			})
			.catch((error) => {
				console.error('Error:', error);
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
			{result && <div>Result: {JSON.stringify(result)}</div>}
		</div>
	);
};

export default SampleDataButton;
