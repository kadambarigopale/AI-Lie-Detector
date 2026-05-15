document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const elements = {
        verdictTitle: document.getElementById('verdictTitle'),
        confidenceText: document.getElementById('confidenceText'),
        confidenceFill: document.getElementById('confidenceFill'),
        panel: document.getElementById('dashboardPanel'),
        sentimentBadge: document.getElementById('sentimentBadge'),
        highlightedText: document.getElementById('highlightedText')
    };

    // Retrieve data from sessionStorage
    const reportDataStr = sessionStorage.getItem('lastAnalysis');
    const originalText = sessionStorage.getItem('lastText');

    if (!reportDataStr) {
        console.error("No report data found. Redirecting to home.");
        window.location.href = 'index.html';
        return;
    }

    const data = JSON.parse(reportDataStr);
    showResults(data, originalText || "");

    function showResults(data, originalText) {
        const isTruthful = (data.prediction || "").toLowerCase() === 'truthful';
        const percent = Math.round((data.confidence || 0) * 100);

        // Update Verdict
        if (elements.verdictTitle) {
            elements.verdictTitle.innerText = isTruthful ? "Truthful Statement" : "Deceptive Statement";
        }
        
        if (elements.confidenceText) {
            elements.confidenceText.innerText = `AI Confidence Level: ${percent}%`;
        }

        // Apply Theme
        if (elements.panel) {
            elements.panel.classList.remove('is-truthful', 'is-deceptive');
            elements.panel.classList.add(isTruthful ? 'is-truthful' : 'is-deceptive');
        }

        // Sentiment
        if (elements.sentimentBadge) {
            const sentiment = data.sentiment || "Neutral";
            elements.sentimentBadge.innerText = sentiment;
            elements.sentimentBadge.className = 'sentiment-badge ' + sentiment.toLowerCase().replace(/\s+/g, '-');
        }

        // Highlighting
        if (elements.highlightedText) {
            highlightText(originalText, data.suspicious_words || []);
        }

        // Animate Bar
        setTimeout(() => {
            if (elements.confidenceFill) elements.confidenceFill.style.width = `${percent}%`;
        }, 300);
    }

    function highlightText(text, suspiciousWords) {
        if (!elements.highlightedText) return;
        
        if (!suspiciousWords || suspiciousWords.length === 0) {
            elements.highlightedText.innerText = text;
            return;
        }

        const escapeHtml = (unsafe) => {
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        };

        let safeText = escapeHtml(text);
        const escapeRegExp = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const wordsRegex = suspiciousWords.map(w => escapeRegExp(w)).join('|');
        const regex = new RegExp(`\\b(${wordsRegex})\\b`, 'gi');

        safeText = safeText.replace(regex, '<mark>$1</mark>');
        elements.highlightedText.innerHTML = safeText;
    }
});
