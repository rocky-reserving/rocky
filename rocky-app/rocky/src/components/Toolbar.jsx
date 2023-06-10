import { useState } from 'react';
import ToolbarIcon from './_toolbar/ToolbarIcon';
import Accordion from './_toolbar/Accordion';
import SaveLoadButtons from './_toolbar/SaveLoadButton';

const Toolbar = () => {
	// state variable to keep track of whether the toolbar is minimized or not
	const [isMinimized, setIsMinimized] = useState(true);

	return (
		// return a div with a conditional class name based on the state variable
		// (minimized or not)
		<div
			className={`bg-blue-200 transition-width duration-300 ease-in-out border-r border-black ${
				isMinimized ? 'w-1/20' : 'w-1/5'
			} h-screen`}
			// set the state variable `IsMinimized` to be false when the mouse enters
			// the div, and true when the mouse leaves the div
			onMouseEnter={() => setIsMinimized(false)}
			onMouseLeave={() => setIsMinimized(true)}
		>
			{/* ======================================== */}
			{/* we are only inside the div starting here */}
			{/* ======================================== */}

			{/* render the ToolbarIcon component */}
			<ToolbarIcon />

			{/* render the Accordion component if the div is not minimized */}
			{/* eg render when the mouse enters the div */}
			{!isMinimized && (
				<>
					{/* render Accordion & SaveLoadButtons */}
					<Accordion />
					<SaveLoadButtons />
				</>
			)}
		</div>
	);
};

export default Toolbar;
