// Explainability visualization utilities

function visualizeFeatureImportance(features, containerId) {
    const container = document.getElementById(containerId);
    
    // Sort features by importance
    const sortedFeatures = Object.entries(features)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);
    
    const maxValue = Math.max(...sortedFeatures.map(([_, val]) => val));
    
    let html = '<div class="feature-importance-chart">';
    
    sortedFeatures.forEach(([feature, value]) => {
        const percentage = (value / maxValue) * 100;
        html += `
            <div class="feature-item">
                <div class="feature-label">${feature}</div>
                <div class="feature-bar-wrapper">
                    <div class="feature-bar-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="feature-value">${value.toFixed(4)}</div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function generateExplanationText(explanation) {
    const { category_name, confidence, top_keywords, reasoning } = explanation;
    
    return `
        <div class="explanation-text">
            <h4>Prediction: ${category_name}</h4>
            <p><strong>Confidence:</strong> ${(confidence * 100).toFixed(2)}%</p>
            <p><strong>Reasoning:</strong> ${reasoning}</p>
            <div class="keywords-section">
                <strong>Key Indicators:</strong>
                ${top_keywords.map(kw => `<span class="keyword-badge">${kw}</span>`).join('')}
            </div>
        </div>
    `;
}

