import React, {useState} from 'react';
import useMoveResize from './useMoveResize';
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
    const panelRef = useRef(null);
    useMoveResize(panelRef);

    return (
        <div className="panel" id={id} ref={panelRef}>
            <div className="resize-handle"></div>
            {children}
        </div>
    );
};




function App() {
  return (
        <div className="App">
            <Toolbar>
                <Accordion title="Data Loading">
                    {/* Accordion content for data loading goes here */}
                </Accordion>
                <Accordion title="Model Selection">
                    {/* Accordion content for model selection goes here */}
                </Accordion>
                <Accordion title="Model Output">
                    {/* Accordion content for model output goes here */}
                </Accordion>
                <Accordion title="Model Diagnostics">
                    {/* Accordion content for model diagnostics goes here */}
                </Accordion>
                {/* Save and Load buttons */}
                <button className="toolbar-button">Save</button>
                <button className="toolbar-button">Load</button>
            </Toolbar>

            <div className="main-area">
                <Panel id="panel1">
                    {/* Panel 1 content goes here */}
                </Panel>
                <Panel id="panel2">
                    {/* Panel 2 content goes here */}
                </Panel>
            </div>
        </div>
    );
}

export default App;
