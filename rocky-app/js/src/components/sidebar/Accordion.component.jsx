import PropTypes from 'prop-types';

const Accordion = ({
	title,
	itemIcon,
	items,
	isSidebarExpanded,
	isActive,
	onToggleAccordion,
	onClickItem,
	onClickNew,
}) => {
	return (
		<div className="accordion">
			<button
				className="accordion-title"
				onClick={title === 'New' ? onClickNew : onToggleAccordion}
			>
				{isSidebarExpanded ? title : itemIcon}
			</button>
			{isActive && (
				<ul className="accordion-content">
					{items.map((item, index) => (
						<li
							key={index}
							onClick={() =>
								item.id === 'new' ? onClickNew : onClickItem(item.title)
							}
						>
							{item.title}
						</li>
					))}
				</ul>
			)}
		</div>
	);
};
Accordion.propTypes = {
	title: PropTypes.string,
	items: PropTypes.arrayOf(
		PropTypes.shape({
			id: PropTypes.string,
			title: PropTypes.string,
			headerText: PropTypes.string,
			divClassName: PropTypes.string,
		}),
	),
	itemIcon: PropTypes.node,
	isSidebarExpanded: PropTypes.bool,
	isActive: PropTypes.bool,
	onToggleAccordion: PropTypes.func,
	onClickItem: PropTypes.func,
	onClickNew: PropTypes.func,
};

export default Accordion;
