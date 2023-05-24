import PropTypes from 'prop-types';

const LandingButton = ({ onClickLoadButton, isFreshlyLoaded }) => {
	return (
		<div className="landing-button">
			{isFreshlyLoaded && (
				<button className="load-data-button" onClick={onClickLoadButton}>
					Get Started
				</button>
			)}
		</div>
	);
};
LandingButton.propTypes = {
	onClickLoadButton: PropTypes.func,
	isFreshlyLoaded: PropTypes.bool,
};

export default LandingButton;
