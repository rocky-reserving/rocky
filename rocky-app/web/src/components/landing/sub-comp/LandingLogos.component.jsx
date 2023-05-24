import reactLogo from '../../.././assets/react.svg';
import viteLogo from '../../../../public/vite.svg';

const LandingLogos = () => {
	return (
		<div className="landing-logos">
			<a className="logo" href="#" target="_blank">
				<img src={viteLogo} alt="Vite logo" />
			</a>
			<a className="logo" href="#" target="_blank">
				<img src={reactLogo} alt="React logo" />
			</a>
		</div>
	);
};

export default LandingLogos;
