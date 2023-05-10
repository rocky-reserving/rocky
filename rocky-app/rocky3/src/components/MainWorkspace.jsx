import LoadDataButton from './LoadDataButton';
import LoadDataWindow from './windows/LoadDataWindow';
import PropTypes from 'prop-types';

const MainWorkspace = ({
	loadDataWindows,
	isFreshlyLoaded,
	onClickLoadButton,
}) => {
	return (
		<div id="main-workspace">
			<div id="load-data-buttons">
				{isFreshlyLoaded && <LoadDataButton onClick={onClickLoadButton} />}
			</div>

			{Object.entries(loadDataWindows).map(([key, value]) => (
				<LoadDataWindow key={key} title={value.title} />
			))}

			<div id="load-data"></div>
			<div id="model-selection"></div>
			<div id="model-validation"></div>
		</div>
	);
};
MainWorkspace.propTypes = {
	loadDataWindows: PropTypes.object,
	isFreshlyLoaded: PropTypes.bool,
	onClickLoadButton: PropTypes.func,
};

export default MainWorkspace;
