import { useState } from 'react';
import Draggable from 'react-draggable';
import { Resizable } from 're-resizable';
import PropTypes from 'prop-types';

const windowTypes = {
	default: 'window-header-default',
	loadData: 'window-header-load-data',
	modelSelection: 'window-header-model-selection',
	modelValidation: 'window-header-model-validation',
};

const BaseDraggableWindow = ({
	children,
	title,
	defautWidth,
	defaultHeight,
	windowType,
	startMinimized,
}) => {
	// state vars
	const [isMinimized, setIsMinimized] = useState(startMinimized);
	const [isFullScreen, setIsFullScreen] = useState(false);
	const [isVisable, setIsVisable] = useState(true);

	const handleMinimize = () => {
		setIsMinimized(!isMinimized);
		setIsFullScreen(false);
	};

	const handleFullScreen = () => {
		setIsFullScreen(!isFullScreen);
		setIsMinimized(false);
	};

	const handleClose = () => {
		setIsVisable(false);
	};

	const renderContent = () => {
		return (
			<Resizable
				defaultSize={{
					width: defautWidth || 300,
					height: defaultHeight || 200,
				}}
				minWidth={300}
				minHeight={200}
			>
				<div className="window-content">
					{!isMinimized && children}{' '}
					{/* Show content when the window is not minimized */}
				</div>
			</Resizable>
		);
	};

	// if the window is not visable, return null instead of the window
	if (!isVisable) {
		return null;
	}

	return (
		<Draggable handle=".window-header">
			<div
				className={`window ${isMinimized ? 'window-closed' : 'window-open'}`}
			>
				<div
					className={`
            window-header
            ${isMinimized ? 'window-header-closed' : 'window-header-open'}
            ${
							windowTypes[windowType] ||
							'window-header-default' /*different colors for different window types*/
						} 
            `}
				>
					<span className="window-title">{title || 'window'}</span>
					<button className="min-max-button" onClick={handleMinimize}>
						{isMinimized ? '+' : '-'}
					</button>
					<button className="fullscreen-button" onClick={handleFullScreen}>
						{isFullScreen ? '↙' : '⤢'}
					</button>
					<button className="close-button" onClick={handleClose}>
						{'×'}
					</button>
				</div>
				{renderContent()}
			</div>
		</Draggable>
	);
};
BaseDraggableWindow.propTypes = {
	children: PropTypes.node,
	title: PropTypes.string,
	defautWidth: PropTypes.number,
	defaultHeight: PropTypes.number,
	startMinimized: PropTypes.bool,
	windowType: PropTypes.string,
};

export default BaseDraggableWindow;
