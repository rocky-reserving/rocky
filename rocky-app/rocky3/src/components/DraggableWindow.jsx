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

const DraggableWindow = ({
	children,
	title,
	defautWidth,
	defaultHeight,
	windowType,
	startMinimized = false,
}) => {
	const [isMinimized, setIsMinimized] = useState(startMinimized);

	const handleMinimize = () => {
		setIsMinimized(!isMinimized);
	};

	const renderContent = () => {
		// if (isMinimized) {
		// 	return null;
		// }

		return (
			<Resizable
				defaultSize={{
					width: defautWidth || 300,
					height: defaultHeight || 200,
				}}
				minWidth={300}
				minHeight={200}
			>
				<div className="window-content">{isMinimized && children}</div>
			</Resizable>
		);
	};

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
				</div>
				{renderContent()}
			</div>
		</Draggable>
	);
};
DraggableWindow.propTypes = {
	children: PropTypes.node,
	title: PropTypes.string,
	defautWidth: PropTypes.number,
	defaultHeight: PropTypes.number,
	startMinimized: PropTypes.bool,
	windowType: PropTypes.string,
};

export default DraggableWindow;
