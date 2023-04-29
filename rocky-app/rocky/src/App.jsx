import Toolbar from './components/Toolbar';
import Workspace from './components/Workspace';

const App = () => {
	return (
		<div className="flex h-screen">
			<Toolbar />
			<Workspace />
		</div>
	);
};

export default App;
