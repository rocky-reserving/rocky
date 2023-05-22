import { DateTime } from 'luxon';

export class Triangle {
	constructor(
		id = null,
		tri = null,
		triangle = null,
		incr_triangle = null,
		X_base = null,
		y_base = null,
		X_base_train = null,
		y_base_train = null,
		X_base_forecast = null,
		y_base_forecast = null,
		has_cum_model_file = false,
		is_cum_model = null,
		frequency = null,
	) {
		this.id = id;
		this.tri = tri;
		this.triangle = triangle;
		this.incr_triangle = incr_triangle;
		this.X_base = X_base;
		this.y_base = y_base;
		this.X_base_train = X_base_train;
		this.y_base_train = y_base_train;
		this.X_base_forecast = X_base_forecast;
		this.y_base_forecast = y_base_forecast;
		this.has_cum_model_file = has_cum_model_file;
		this.is_cum_model = is_cum_model;
		this.frequency = frequency;

		// Call the postInit method
		this.postInit();
	}

	postInit() {
		// Reformat the id
		if (this.id !== null) {
			this.id = this.id.toLowerCase();
			this.id.replace(' ', '_');

			// Reset the id if it is not allowed
			if (!triangle_type_aliases.includes(this.id)) {
				this.id = null;
			}
		}

		// If a triangle was passed in
		if (this.tri !== null) {
			// Set the n_rows and n_cols attributes
			this.n_rows = this.tri.length;
			this.n_cols = this.tri[0].length;

			// Convert the origin to a datetime object
			this.convertOriginToDatetime();

			// Convert triangle data to float
			this.convertTriangleDataToFloat();
		}
	}

	convertTriangleDataToFloat() {
		// Convert triangle data to float if it is not null
		if (this.tri !== null) {
			for (let c = 0; c < this.tri[0].length; c++) {
				try {
					this.tri = this.tri.map((row) => {
						row[c] = parseFloat(row[c]);
						return row;
					});
				} catch (error) {
					this.tri = this.tri.map((row) => {
						row[c] = parseFloat(
							row[c].replace(',', '').replace(')', '').replace('(', '-'),
						);
						return row;
					});
				}
			}
		}
	}

	convertOriginToDatetime() {
		this.tri = this.tri.map((row) => {
			const origin = row.origin;

			// Helper function to convert year, month, and day to a DateTime object
			function convertToDateTime(year, month, day = 1) {
				return DateTime.fromObject({ year, month, day });
			}

			// Helper function to process the origin with a specific delimiter
			function processOriginWithDelimiter(origin, delimiter) {
				const parts = origin.split(delimiter);
				let year, month;

				if (parts.length === 2) {
					const a = parseInt(parts[0], 10);
					const b = parseInt(parts[1], 10);

					if (a && b) {
						if (1000 <= a && a <= 9999 && 1 <= b && b <= 12) {
							year = a;
							month = b;
						} else if (1000 <= b && b <= 9999 && 1 <= a && a <= 12) {
							year = b;
							month = a;
						}
					}
				}

				return { year, month };
			}

			// Check if the origin is an integer
			const intValue = parseInt(origin, 10);
			if (!isNaN(intValue)) {
				if (1000 <= intValue && intValue <= 9999) {
					return { ...row, origin: convertToDateTime(intValue, 1) };
				} else if (0 <= intValue && intValue <= 99) {
					return { ...row, origin: convertToDateTime(intValue + 2000, 1) };
				}
			}

			// Check if the origin contains '-' or '/'
			if (origin.includes('-') || origin.includes('/')) {
				const delimiter = origin.includes('-') ? '-' : '/';
				const { year, month } = processOriginWithDelimiter(origin, delimiter);

				if (year && month) {
					return { ...row, origin: convertToDateTime(year, month) };
				}
			}

			// Check if the origin contains 'Q' or 'q'
			if (origin.toUpperCase().includes('Q')) {
				const parts = origin.toUpperCase().split('Q');
				if (parts.length === 2) {
					const a = parseInt(parts[0], 10);
					const b = parseInt(parts[1], 10);
					let year, quarter;

					if (a && b) {
						if (1000 <= a && a <= 9999 && 1 <= b && b <= 4) {
							year = a;
							quarter = b;
						} else if (1000 <= b && b <= 9999 && 1 <= a && a <= 4) {
							year = b;
							quarter = a;
						}

						if (year && quarter) {
							const month = quarter * 3 - 2;
							return { ...row, origin: convertToDateTime(year, month) };
						}
					}
				}
			}

			throw new Error('Invalid origin column');
		});
	}
}

// allowed triangle types
const triangle_type_aliases = [
	'paid_loss',
	'reported_loss',
	'rpt_loss',
	'case_reserve',
	'paid_dcce',
	'paid_alae',
	'paid_expense',
	'reported',
];
