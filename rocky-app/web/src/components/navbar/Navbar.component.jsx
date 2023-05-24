import NavbarLogo from './sub-comp/NavbarLogo.component';
import NavbarSeparator from './sub-comp/NavbarSeparator.component';
import NavbarMenu from './sub-comp/NavbarMenu.component';

const Navbar = () => {
	return (
		<nav className="navbar">
			<NavbarLogo />
			<NavbarSeparator />
			<NavbarMenu />
		</nav>
	);
};

export default Navbar;
