import LandingLogos from './sub-comp/LandingLogos.component';
import LandingButton from './sub-comp/LandingButton.component';
import LandingHeader from './sub-comp/LandingHeader.component';
import LandingMain from './sub-comp/LandingMain.component';

import '../../App.css';

import PropTypes from 'prop-types';

const Landing = ({ onClickLoadButton, isFreshlyLoaded, isSidebarExpanded }) => {
	return (
		<div className={`landing ${isSidebarExpanded ? 'expanded' : 'collapsed'}`}>
			<LandingLogos />
			<LandingHeader />
			<LandingButton
				onClickLoadButton={onClickLoadButton}
				isFreshlyLoaded={isFreshlyLoaded}
			/>
			<LandingMain />
		</div>
	);
};
Landing.propTypes = {
	onClickLoadButton: PropTypes.func,
	isFreshlyLoaded: PropTypes.bool,
	isSidebarExpanded: PropTypes.bool,
};

export default Landing;
