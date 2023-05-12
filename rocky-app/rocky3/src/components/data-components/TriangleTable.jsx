import PropTypes from 'prop-types';
import styles from './TriangleTable.module.css';

const formatDate = (timestamp) => {
	if (isNaN(timestamp)) {
		return timestamp;
	}

	const date = new Date(parseInt(timestamp));
	return date.toLocaleDateString();
};

const formatNumberWithCommas = (number) => {
	if (number === null || number === undefined) {
		return '';
	}
	return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
};

const TriangleTable = ({ data }) => {
	if (!data) {
		return <div>No data available.</div>;
	}

	const rowKeys = Object.keys(data);
	const colKeys = Object.keys(data[rowKeys[0]]);

	const formattedColKeys = colKeys.map((colKey) => formatDate(colKey));

	return (
		<table className={styles['triangle-table']}>
			<thead>
				{/* <span className="triangle-header"> */}
				<tr>
					<th></th>
					{rowKeys.map((rowKey, index) => (
						<th key={index}>{data[rowKey].id}</th>
					))}
				</tr>
				{/* </span> */}
			</thead>
			<tbody>
				{formattedColKeys.slice(0, -1).map((colKey, colIndex) => (
					<tr key={colIndex}>
						<td>{colKey}</td>
						{rowKeys.map((rowKey, rowIndex) => (
							<td key={rowIndex}>
								{formatNumberWithCommas(data[rowKey][colKeys[colIndex]])}
							</td>
						))}
					</tr>
				))}
			</tbody>
		</table>
	);
};

TriangleTable.propTypes = {
	data: PropTypes.objectOf(PropTypes.object),
};

export default TriangleTable;
