import LoadDataWindow from './load-data-window/LoadDataWindow.component';
import PropTypes from 'prop-types';

const MainWorkspace = ({
	loadDataWindows,
	triangleParentSize,
	triangleRef,
}) => {
	return (
		<div id="main-workspace">
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
	triangleParentSize: PropTypes.object,
	triangleRef: PropTypes.object,
};

export default MainWorkspace;
