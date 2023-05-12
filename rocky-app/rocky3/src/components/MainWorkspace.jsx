import LoadDataButton from './buttons/LoadDataButton';
import LoadDataWindow from './windows/LoadDataWindow';
import PropTypes from 'prop-types';

const MainWorkspace = ({
	loadDataWindows,
	isFreshlyLoaded,
	onClickLoadButton,
	triangleParentSize,
	triangleRef,
}) => {
	return (
		<div id="main-workspace">
			<div id="load-data-buttons">
				{isFreshlyLoaded && <LoadDataButton onClick={onClickLoadButton} />}
			</div>

			{Object.entries(loadDataWindows).map(([key, value]) => (
				<LoadDataWindow
					key={key}
					title={value.title}
					triangleParentSize={triangleParentSize}
					triangleRef={triangleRef}
				/>
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
	triangleParentSize: PropTypes.object,
	triangleRef: PropTypes.object,
};

export default MainWorkspace;
