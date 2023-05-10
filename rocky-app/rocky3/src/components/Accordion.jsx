import PropTypes from 'prop-types';

const Accordion = ({
	title,
	itemIcon,
	items,
	isSidebarExpanded,
	isActive,
	onToggleAccordion,
}) => {
	return (
		<div className="accordion">
			<button className="accordion-title" onClick={onToggleAccordion}>
				{isSidebarExpanded ? title : itemIcon}
			</button>
			{isActive && (
				<ul className="accordion-content">
					{items.map((item, index) => (
						<li key={index}>{item}</li>
					))}
				</ul>
			)}
		</div>
	);
};
Accordion.propTypes = {
	title: PropTypes.string,
	items: PropTypes.arrayOf(PropTypes.string),
	itemIcon: PropTypes.node,
	isSidebarExpanded: PropTypes.bool,
	isActive: PropTypes.bool,
	onToggleAccordion: PropTypes.func,
};

export default Accordion;
