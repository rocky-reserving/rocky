import PropTypes from 'prop-types';

import appData from '../../../appdata';

import Accordion from './Accordion.component';

const SidebarMenu = ({
	activeAccordion,
	toggleAccordion,
	handleClickItem,
	isSidebarExpanded,
	onAddLoadDataWindow,
	onClickNew,
}) => {
	// load accordion items from appData
	const sidebarItems = appData.sidebarItems;

	return (
		<div className="sidebar-menu">
			{sidebarItems.map((item, index) => (
				<Accordion
					key={index}
					itemID={index}
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
SidebarMenu.propTypes = {
	activeAccordion: PropTypes.number,
	toggleAccordion: PropTypes.func,
	handleClickItem: PropTypes.func,
	isSidebarExpanded: PropTypes.bool,
	onAddLoadDataWindow: PropTypes.func,
	onClickNew: PropTypes.func,
};

export default SidebarMenu;
