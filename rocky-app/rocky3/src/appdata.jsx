import { GrNewWindow } from 'react-icons/gr';
import { BiScatterChart } from 'react-icons/bi';
import { ImFolderUpload } from 'react-icons/im';

// const port = 5000;
const api_url = (func) => {
	let url = `http://0.0.0.0:5000/rockyapi/${func}`;
	return url;
	// console.log('func:', func);
	// console.log('url:', url);
};

const appData = {
	// Sidebar top-level items and their sub-items
	sidebarItems: [
		{
			id: 'new',
			title: 'New',
			itemIcon: <GrNewWindow />,
			expands: false,
			items: [],
		},
		{
			id: 'load-data',
			title: 'Load Data',
			itemIcon: <ImFolderUpload />,
			expands: true,
			items: [
				// sub-categories under the load data top-level item
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
				{
					id: 'sample-data',
					title: 'Sample Data',
					headerText: 'Load Sample Triangle Data',
					divClassName: 'load-sample-data-window',
				},
			],
		},
		{
			id: 'model-selection',
			title: 'Model Selection',
			itemIcon: <BiScatterChart />,
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

	accordionItems: [
		{
			title: 'New',
			itemIcon: <GrNewWindow />,
			items: [],
		},
		{
			title: 'Load Data',
			itemIcon: <ImFolderUpload />,
			items: ['Sample Data', 'Clipboard', 'Excel', 'CSV'],
		},
		{
			title: 'Model Selection',
			itemIcon: <BiScatterChart />,
			items: ['Chain Ladder', 'GLM', 'MegaModel'],
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

	// api endpoints
	api: {
		'Taylor-Ashe Paid Loss': api_url('load-taylor-ashe'),
		'Dahms Reported Loss': api_url('load-dahms-rpt'),
		'Dahms Paid Loss': api_url('load-dahms-paid'),
	},

	// Windows
	// 'window-types': {
	// 	''
};

export default appData;
