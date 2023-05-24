import PropTypes from 'prop-types';

import appData from '../../../appdata';

import AccordionItem from './sub-components/AccordionItem.component';
import AccordionTitle from './sub-components/AccordionTitle.component';

const Accordion = ({
	// itemID,
	isSidebarExpanded,
	isActive,
	onToggleAccordion,
	onClickItem,
	onClickNew,
}) => {
	const log = (text, item) => {
		console.log(text, item);
		return true;
	};

	const sidebarItems = appData.sidebarItems;
	return (
		<div className="accordion">
			{log('sidebarItems: ', sidebarItems) && (
				sidebarItems.map(
					(item, index) => {
						(
							log('item: ', item) && (
							log('item.items: ', item.items)) && (
								<AccordionTitle
									title={item.title}
									itemIcon={item.itemIcon}
									onClickNew={onClickNew}
									onToggleAccordion={onToggleAccordion}
									isSidebarExpanded={isSidebarExpanded}
								/>
							)
						)
						{
							isActive && item.items && (
								log('item: ', item)) && (
									log('item.items: ', item.items)) && (
									item.items.map((item2, index2) => (
										<AccordionItem
											key={`${index2}`}
											isSidebarExpanded={isSidebarExpanded}
											onToggleAccordion={onToggleAccordion}
											item={item2}
											onClickNew={onClickNew}
											onClickItem={onClickItem}
											menuItemID={item2.id}
										/>
									)
								)
							)
						}
						
					},
				),
			)
		}

		</div>
	);
};
Accordion.propTypes = {
	itemID: PropTypes.number,
	isSidebarExpanded: PropTypes.bool,
	isActive: PropTypes.bool,
	onToggleAccordion: PropTypes.func,
	onClickItem: PropTypes.func,
	onClickNew: PropTypes.func,
};

export default Accordion;
