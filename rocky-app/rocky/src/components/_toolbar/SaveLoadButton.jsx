// import React from 'react';

const SaveLoadButtons = () => {
	const handleSave = () => {
		// Save the state of the web app to a JSON file
		alert(
			"I may not have actually done anything, but I'm still a button!\n\nAdd the `handleSave` function to `SaveLoadButtons.jsx`",
		);
	};

	const handleLoad = () => {
		// Load the state of the web app from a JSON file
		alert(
			"I may not have actually done anything, but I'm still a button!\n\nAdd the `handleLoad` function to `SaveLoadButtons.jsx`",
		);
	};

	return (
		<div className="flex justify-between p-4">
			<button
				className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
				onClick={handleSave}
			>
				Save
			</button>
			<button
				className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
				onClick={handleLoad}
			>
				Load
			</button>
		</div>
	);
};

export default SaveLoadButtons;
