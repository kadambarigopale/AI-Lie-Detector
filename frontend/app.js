const initApp = () => {
    console.log("AI Lie Detector: Initializing...");

    const form = document.getElementById('analysisForm');
    const textIn = document.getElementById('textIn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.querySelector('.btn-text');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (!form || !analyzeBtn) {
        console.error("Critical Error: Analysis elements not found.");
        return;
    }

    // Determine API URL: Use relative if on port 8000 (FastAPI), otherwise absolute localhost:8000
    const API_URL = (window.location.port === '8000') ? '/predict' : 'http://localhost:8000/predict';

    form.onsubmit = async (e) => {
        e.preventDefault();
        
        const text = textIn ? textIn.value.trim() : "";
        if (!text) return;

        // Start Loading
        analyzeBtn.disabled = true;
        if (btnText) btnText.classList.add('hidden');
        if (loadingSpinner) loadingSpinner.classList.remove('hidden');

        console.log("Sending analysis request to:", API_URL);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown server error" }));
                throw new Error(errorData.detail || "Server failed");
            }

            const data = await response.json();
            
            // Success: Save and Redirect
            sessionStorage.setItem('lastAnalysis', JSON.stringify(data));
            sessionStorage.setItem('lastText', text);
            
            console.log("Analysis successful. Redirecting to dashboard...");
            window.location.assign('dashboard.html');

        } catch (error) {
            console.error("Analysis Error:", error);
            alert(`Analysis failed: ${error.message}\n\nPlease ensure your backend is running.`);
            
            // Reset UI
            analyzeBtn.disabled = false;
            if (btnText) btnText.classList.remove('hidden');
            if (loadingSpinner) loadingSpinner.classList.add('hidden');
        }
    };
};

// Start initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
