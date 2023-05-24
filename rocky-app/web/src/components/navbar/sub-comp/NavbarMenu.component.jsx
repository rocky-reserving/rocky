import appData from '../../../appdata';

const NavbarMenu = () => {
	// click handler for each menu item
	// const handleClick = (url) => {
	// 	window.open(url, '_blank');
	// };

	// get navbar items from appdata
	const navItems = appData.navbar;

	// return navbar menu -- ul > li > button
	return (
		<ul className="navbar-menu">
			{navItems.map((item) => {
				return (
					<li key={item.id} className="navbar-menu-item">
						<a href={item.url} name={item.name}>
							{item.name}
						</a>
						{/* <button onClick={() => handleClick(item.url)}>{item.name}</button> */}
					</li>
				);
			})}
		</ul>
	);
};

export default NavbarMenu;
