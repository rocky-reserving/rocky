import { useState } from 'react';
import Accordion from './Accordion';
import { ImFolderUpload } from 'react-icons/im';
import { BiScatterChart } from 'react-icons/bi';

const Sidebar = () => {
	const accordionItems = [
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

	const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);
	const [activeAccordion, setActiveAccordion] = useState(null);

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

	return (
		<div className={`sidebar ${isSidebarExpanded ? 'expanded' : 'collapsed'}`}>
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
				/>
			))}
		</div>
	);
};

export default Sidebar;
