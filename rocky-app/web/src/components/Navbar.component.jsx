import appData from '../appdata';
import { RxDividerVertical } from 'react-icons/rx';

const Navbar = () => {
	const navItems = appData.navbar;

	const handleClick = (url) => {
		window.open(url, '_blank');
	};

	return (
		<nav className="navbar">
			<a href="/" className="navbar-logo">
				Rocky
			</a>
			{/* add a vertical separator here: */}
			<div className="navbar-separator">
				<RxDividerVertical />
			</div>
			<ul className="navbar-menu">
				{navItems.map((item) => {
					return (
						<li key={item.id} className="navbar-menu-item">
							<button onClick={() => handleClick(item.url)}>{item.name}</button>
						</li>
					);
				})}
			</ul>
		</nav>
	);
};

export default Navbar;
