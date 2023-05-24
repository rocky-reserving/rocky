import appData from '../../../../appdata';

import PropTypes from 'prop-types';

const AccordionTitle = ({
	onClickNew,
	onToggleAccordion,
	isSidebarExpanded,
}) => {
	const { title, itemIcon } = appData.sidebarItems[0];
	return (
		<button
			className="accordion-title"
			onClick={title === 'New' ? onClickNew : onToggleAccordion}
		>
			{isSidebarExpanded ? title : itemIcon}
		</button>
	);
};
AccordionTitle.propTypes = {
	onClickNew: PropTypes.func,
	onToggleAccordion: PropTypes.func,
	isSidebarExpanded: PropTypes.bool,
};

export default AccordionTitle;
