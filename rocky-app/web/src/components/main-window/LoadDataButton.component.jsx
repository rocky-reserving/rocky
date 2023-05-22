import PropTypes from 'prop-types';

const LoadDataButton = ({ onClick }) => {
	return (
		<button className="load-data-button" onClick={onClick}>
			Get Started
		</button>
	);
};
LoadDataButton.propTypes = {
	onClick: PropTypes.func,
};

export default LoadDataButton;
