const appData = {
	// Sidebar top-level items and their sub-items
	sidebarItems: [
		{
			id: 'new',
			title: 'New',
			expands: false,
			items: [],
		},
		{
			id: 'load-data',
			title: 'Load Data',
			expands: true,
			items: [
				// sub-categories under the load data top-level item
				{
					id: 'sample-data',
					title: 'Sample Data',
					headerText: 'Load Sample Triangle Data',
					divClassName: 'load-sample-data-window',
				},
				{
					id: 'clipboard',
					title: 'Clipboard',
					headerText: 'Paste Triangle Data from Clipboard',
					divClassName: 'load-clipboard-window',
				},
				{
					id: 'excel',
					title: 'Excel',
					headerText: 'Load Triangle Data from Excel File',
					divClassName: 'load-excel-window',
				},
				{
					id: 'csv',
					title: 'CSV',
					headerText: 'Load Triangle Data from CSV File',
					divClassName: 'load-csv-window',
				},
			],
		},
		{
			id: 'model-selection',
			title: 'Model Selection',
			expands: true,
			items: [
				{
					id: 'chain-ladder',
					title: 'Chain Ladder',
					headerText: 'Chain Ladder Method',
				},
				{
					id: 'glm',
					title: 'GLM',
					headerText: 'GLM Development Trend Model',
				},
				{
					id: 'megamodel',
					title: 'MegaModel',
					headerText: 'MegaModel - The Ultimate Model',
				},
			],
		},
	],

	// available sample data sets
	sampleData: [
		{
			id: 'taylor-ashe',
			name: 'Taylor-Ashe Paid Loss',
		},
		{
			id: 'dahms-rpt',
			name: 'Dahms Reported Loss',
		},
		{
			id: 'dahms-paid',
			name: 'Dahms Paid Loss',
		},
	],

	// Windows
	// 'window-types': {
	// 	''
};

export default appData;
