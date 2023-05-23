import PropTypes from 'prop-types';

const LandingButton = ({ onClick }) => {
	return (
		<div className="landing-button">
			<button className="load-data-button" onClick={onClick}>
				Get Started
			</button>
		</div>
	);
};
LandingButton.propTypes = {
	onClick: PropTypes.func,
};

export default LandingButton;
