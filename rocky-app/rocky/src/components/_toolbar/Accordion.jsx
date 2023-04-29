import { useState } from 'react';
import acc_data from './accordion_data.json';
import PropTypes from 'prop-types';

// component for each child of the accordion
const AccordionChild = ({ child }) => {
	const [isGlowing, setIsGlowing] = useState(false);

	return (
		<div
			className={`px-4 py-2 bg-blue-200 transition duration-300 ease-in-out ${
				isGlowing ? 'drop-shadow-lg' : ''
			}`}
			onMouseEnter={() => setIsGlowing(true)}
			onMouseLeave={() => setIsGlowing(false)}
		>
			{child}
		</div>
	);
};
AccordionChild.propTypes = {
	child: PropTypes.node.isRequired,
};

// component for each accordion item
const AccordionItem = ({ title, item }) => {
	const [isOpen, setIsOpen] = useState(false);

	return (
		<div>
			<button
				className="w-full text-left px-4 py-2 text-lg font-semibold bg-blue-300 hover:bg-blue-400 focus:outline-none"
				onClick={() => setIsOpen(!isOpen)}
			>
				{title}
			</button>
			{isOpen && <AccordionChild child={item.children} />}
		</div>
	);
};
AccordionItem.propTypes = {
	title: PropTypes.string.isRequired,
	item: PropTypes.node.isRequired,
};

const Accordion = () => {
	return (
		<div>
			{acc_data.map((x, index) => (
				<AccordionItem key={index} title={x.title} item={x} />
			))}
		</div>
	);
};

export default Accordion;

// import { useState } from 'react';
// import acc_data from './accordion_data.json';
// import PropTypes from 'prop-types';

// // component for each child of the accordion
// const AccordionChild = ({ child }) => {
// 	// glow when hovered over, so need a state variable to keep track of whether
// 	// the child is glowing or not
// 	const [isGlowing, setIsGlowing] = useState(false);

// 	return (
// 		<div
// 			className="px-4 py-2 bg-gray-200 hover:drop-shadow-lg transition duration-300 ease-in-out"
// 			onMouseEnter={() => setIsGlowing(true)}
// 			onMouseLeave={() => setIsGlowing(false)}
// 		>
// 			{child}
// 		</div>
// 	);
// };
// AccordionChild.propTypes = {
// 	child: PropTypes.node.isRequired,
// };

// // component for each accordion item
// const AccordionItem = ({ title, item }) => {
// 	// state variable to keep track of whether the accordion item is open or not
// 	const [isOpen, setIsOpen] = useState(false);

// 	return (
// 		<div>
// 			<button
// 				className="w-full text-left px-4 py-2 text-lg font-semibold bg-gray-300 hover:bg-gray-400 focus:outline-none"
// 				onClick={() => setIsOpen(!isOpen)}
// 			>
// 				{title}
// 			</button>
// 			if (isOpen){<AccordionChild child={item.children} />}
// 			{/* {isOpen && <div className="px-4 py-2">{props.children}</div>} */}
// 		</div>
// 	);
// };
// AccordionItem.propTypes = {
// 	title: PropTypes.string.isRequired,
// 	item: PropTypes.node.isRequired,
// };

// const Accordion = () => {
// 	return (
// 		<div>
// 			{acc_data.map((x) => (
// 				<AccordionItem title={x.title} item={x} />
// 			))}
// 		</div>
// 	);
// };
// Accordion.propTypes = {};

// export default Accordion;
