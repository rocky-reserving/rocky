import PropTypes from 'prop-types';

const SidebarToggle = ({ toggleSidebar, isSidebarExpanded }) => {
	return (
		<button className="sidebar-toggle" onClick={toggleSidebar}>
			{isSidebarExpanded ? '<' : '>'}
		</button>
	);
};

SidebarToggle.propTypes = {
	toggleSidebar: PropTypes.func,
	isSidebarExpanded: PropTypes.bool,
};

export default SidebarToggle;
