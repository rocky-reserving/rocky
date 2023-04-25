import React from 'react';
import './styles.css';


import logo from './logo.svg';
import './App.css';

const Toolbar = ({ children }) => {
    // Toolbar logic
  
    return (
        <div className="toolbar" /* ... */>
            {children}
        </div>
    );
};

const Accordion = ({ title, children }) => {
    // Accordion logic
  
    return (
        <details className="accordion" /* ... */>
            <summary className="accordion-summary">
                {/* Add icon here */} 
                {title}
            </summary>
            <div className="accordion-content">
                {children}
            </div>
        </details>
    );
};

const Panel = ({ id, children }) => {
    // Panel logic

    return (
        <div className="panel" id={id} /* ... */>
            <div className="resize-handle"></div>
            {children}
        </div>
    );
};



function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
            
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
