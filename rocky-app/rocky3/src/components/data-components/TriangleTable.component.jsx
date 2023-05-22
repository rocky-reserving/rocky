import PropTypes from 'prop-types';
import styles from './TriangleTable.styles.css';

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
	if (!data || data.length === 0) {
		return <div>No data available.</div>;
	} else {
		const rowKeys = data.map((row) => row.id);
		const colKeys = Object.keys(data[0]).filter((key) => key !== 'id');
		const formattedColKeys = colKeys.map((colKey) => formatDate(colKey));

		return (
			<table className={styles['triangle-table']}>
				<thead>
					<tr>
						<th></th>
						{rowKeys.map((rowKey, index) => (
							<th key={index}>{rowKey}</th>
						))}
					</tr>
				</thead>
				<tbody>
					{formattedColKeys.slice(0, -1).map((colKey, colIndex) => (
						<tr key={colIndex}>
							<td>{colKey}</td>
							{rowKeys.map((rowKey, rowIndex) => (
								<td key={rowIndex}>
									{formatNumberWithCommas(data[rowIndex][colKeys[colIndex]])}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>
		);
	}
};

TriangleTable.propTypes = {
	data: PropTypes.arrayOf(PropTypes.object),
};

export default TriangleTable;
