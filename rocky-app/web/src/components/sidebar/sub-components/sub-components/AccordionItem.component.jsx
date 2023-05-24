import PropTypes from 'prop-types';

const AccordionItem = ({ item, menuItemID, onClickNew, onClickItem }) => {
	const log = (item) => {
		console.log(item);
		return true;
	};
	return (
		<div className="accordion-item">
			<ul className="accordion-content">
				{item.items.map((item, index) => (
					<li
						key={index}
						onClick={() =>
							menuItemID === 'new' ? onClickNew : onClickItem(item.title)
						}
					>
						{log(item.title) && item.title}
					</li>
				))}
			</ul>
		</div>
	);
};
AccordionItem.propTypes = {
	item: PropTypes.object,
	onClickNew: PropTypes.func,
	onClickItem: PropTypes.func,
	menuItemID: PropTypes.string,
};

export default AccordionItem;
