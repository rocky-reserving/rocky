import { useState, useEffect } from 'react';
import Accordion from './Accordion';
import { ImFolderUpload } from 'react-icons/im';
import { BiScatterChart } from 'react-icons/bi';
import { GrNewWindow } from 'react-icons/gr';
import PropTypes from 'prop-types';

const Sidebar = ({
	isSidebarExpanded,
	setIsSidebarExpanded,
	onAddLoadDataWindow,
	onClickNew,
}) => {
	const accordionItems = [
		{
			title: 'New',
			itemIcon: <GrNewWindow />,
			items: [],
		},
		{
			title: 'Load Data',
			itemIcon: <ImFolderUpload />,
			items: ['Sample Data', 'Clipboard', 'Excel', 'CSV'],
		},
		{
			title: 'Model Selection',
			itemIcon: <BiScatterChart />,
			items: ['Chain Ladder', 'GLM', 'MegaModel'],
		},
	];

	const [activeAccordion, setActiveAccordion] = useState(null);
	const [timeoutId, setTimeoutId] = useState(null);

	const startCountdown = () => {
		if (isSidebarExpanded) {
			const id = setTimeout(() => {
				setIsSidebarExpanded(false);
				setActiveAccordion(null);
			}, 2000);
			setTimeoutId(id);
		}
	};

	const cancelCountdown = () => {
		if (timeoutId) {
			clearTimeout(timeoutId);
			setTimeoutId(null);
		}
	};

	useEffect(() => {
		return () => {
			if (timeoutId) {
				clearTimeout(timeoutId);
			}
		};
	}, [timeoutId]);

	const toggleSidebar = () => {
		setIsSidebarExpanded(!isSidebarExpanded);
		setActiveAccordion(null);
	};

	const toggleAccordion = (index) => {
		if (activeAccordion === index) {
			setActiveAccordion(null);
		} else {
			setActiveAccordion(index);
			setIsSidebarExpanded(true);
		}
	};

	const handleClickItem = (item) => {
		// NEW
		if (item === 'New') {
			onClickNew();
		}

		// LOAD DATA
		else if (item === 'Sample Data') {
			onAddLoadDataWindow('Sample Data');
		} else if (item === 'Clipboard') {
			onAddLoadDataWindow('Clipboard');
		} else if (item === 'Excel') {
			onAddLoadDataWindow('Excel');
		} else if (item === 'CSV') {
			onAddLoadDataWindow('CSV');
		}

		// MODEL SELECTION
		// else if (item === 'Chain Ladder') { }
		// else if (item === 'GLM') { }
		// else if (item === 'MegaModel') { }

		// Add conditions for other items here
	};

	return (
		<div
			className={`sidebar ${isSidebarExpanded ? 'expanded' : 'collapsed'}`}
			onMouseLeave={startCountdown}
			onMouseEnter={cancelCountdown}
		>
			<button className="sidebar-toggle" onClick={toggleSidebar}>
				{isSidebarExpanded ? '<' : '>'}
			</button>
			{accordionItems.map((item, index) => (
				<Accordion
					key={index}
					title={item.title}
					itemIcon={item.itemIcon}
					items={item.items}
					isSidebarExpanded={isSidebarExpanded}
					isActive={activeAccordion === index}
					onToggleAccordion={() => toggleAccordion(index)}
					onAddLoadDataWindow={onAddLoadDataWindow}
					onClickNew={onClickNew}
					onClickItem={handleClickItem}
				/>
			))}
		</div>
	);
};
Sidebar.propTypes = {
	isSidebarExpanded: PropTypes.bool,
	setIsSidebarExpanded: PropTypes.func,
	onAddLoadDataWindow: PropTypes.func,
	onClickNew: PropTypes.func,
	onToggleAccordion: PropTypes.func,
	onClickItem: PropTypes.func,
};

export default Sidebar;
