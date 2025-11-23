const API_BASE_URL = 'http://localhost:8000';

// Tab management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Activate button
    event.target.classList.add('active');
}

// Categorize single transaction
async function categorizeTransaction() {
    const transactionString = document.getElementById('transaction-input').value.trim();
    const amount = parseFloat(document.getElementById('amount-input').value) || null;
    
    if (!transactionString) {
        showError('categorize-result', 'Please enter a transaction string');
        return;
    }
    
    showLoading('categorize-result');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/categorize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                transaction_string: transactionString,
                amount: amount
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayCategorizeResult(data);
    } catch (error) {
        showError('categorize-result', `Error: ${error.message}`);
    }
}

function displayCategorizeResult(data) {
    const resultDiv = document.getElementById('categorize-result');
    const confidenceClass = getConfidenceClass(data.confidence_score);
    
    resultDiv.innerHTML = `
        <div class="result-item">
            <h3>Transaction: ${data.transaction_string}</h3>
            <p><strong>Predicted Category:</strong> ${data.predicted_category}</p>
            <p><strong>Confidence Score:</strong> 
                <span class="${confidenceClass}">${(data.confidence_score * 100).toFixed(2)}%</span>
            </p>
            ${data.amount ? `<p><strong>Amount:</strong> $${data.amount.toFixed(2)}</p>` : ''}
            <p><strong>Transaction ID:</strong> ${data.id}</p>
            <button onclick="getExplanationForTransaction(${data.id})" class="btn btn-secondary" style="margin-top: 10px;">
                Get Explanation
            </button>
        </div>
    `;
    resultDiv.classList.add('show');
}

// Batch categorization
async function categorizeBatch() {
    const batchInput = document.getElementById('batch-input').value.trim();
    
    if (!batchInput) {
        showError('batch-result', 'Please enter transaction strings');
        return;
    }
    
    const transactions = batchInput.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .map(line => ({ transaction_string: line }));
    
    if (transactions.length === 0) {
        showError('batch-result', 'No valid transactions found');
        return;
    }
    
    showLoading('batch-result');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/categorize/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transactions })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayBatchResult(data.results);
    } catch (error) {
        showError('batch-result', `Error: ${error.message}`);
    }
}

function displayBatchResult(results) {
    const resultDiv = document.getElementById('batch-result');
    
    let html = `<h3>Batch Results (${results.length} transactions)</h3>`;
    
    results.forEach(result => {
        const confidenceClass = getConfidenceClass(result.confidence_score);
        html += `
            <div class="result-item">
                <p><strong>${result.transaction_string}</strong></p>
                <p>Category: ${result.predicted_category} | 
                   Confidence: <span class="${confidenceClass}">${(result.confidence_score * 100).toFixed(2)}%</span>
                </p>
            </div>
        `;
    });
    
    resultDiv.innerHTML = html;
    resultDiv.classList.add('show');
}

// Get explanation
async function getExplanation() {
    const transactionId = parseInt(document.getElementById('explain-transaction-id').value);
    
    if (!transactionId) {
        showError('explain-result', 'Please enter a transaction ID');
        return;
    }
    
    showLoading('explain-result');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/explain/${transactionId}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayExplanation(data);
    } catch (error) {
        showError('explain-result', `Error: ${error.message}`);
    }
}

async function getExplanationForTransaction(transactionId) {
    document.getElementById('explain-transaction-id').value = transactionId;
    showTab('explain');
    document.querySelectorAll('.tab-button').forEach(btn => {
        if (btn.textContent === 'Explainability') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    await getExplanation();
}

function displayExplanation(data) {
    const resultDiv = document.getElementById('explain-result');
    
    // Sort features by importance
    const features = Object.entries(data.feature_attributions)
        .sort((a, b) => b[1] - a[1]);
    
    let html = `
        <div class="explanation-box">
            <h3>Explanation for Transaction #${data.transaction_id}</h3>
            <p><strong>Reasoning:</strong> ${data.reasoning}</p>
            
            <h4 style="margin-top: 20px;">Top Keywords:</h4>
            <div>
                ${data.top_keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
            </div>
            
            <h4 style="margin-top: 20px;">Feature Attributions:</h4>
            <div>
                ${features.slice(0, 10).map(([feature, value]) => `
                    <div class="feature-bar">
                        <div class="feature-name">${feature}</div>
                        <div class="feature-bar-container">
                            <div class="feature-bar-fill" style="width: ${(value * 100).toFixed(1)}%"></div>
                        </div>
                        <div class="feature-value">${value.toFixed(4)}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.classList.add('show');
}

// Submit feedback
async function submitFeedback() {
    const transactionId = parseInt(document.getElementById('feedback-transaction-id').value);
    const correctedCategory = document.getElementById('feedback-category').value;
    const feedbackReason = document.getElementById('feedback-reason').value;
    
    if (!transactionId || !correctedCategory) {
        showError('feedback-result', 'Please fill in all required fields');
        return;
    }
    
    showLoading('feedback-result');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                transaction_id: transactionId,
                corrected_category: correctedCategory,
                feedback_reason: feedbackReason
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        showSuccess('feedback-result', 'Feedback submitted successfully! Thank you for improving the model.');
        
        // Clear form
        document.getElementById('feedback-transaction-id').value = '';
        document.getElementById('feedback-reason').value = '';
    } catch (error) {
        showError('feedback-result', `Error: ${error.message}`);
    }
}

// Load metrics
async function loadMetrics() {
    showLoading('metrics-result');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/evaluate`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayMetrics(data);
    } catch (error) {
        showError('metrics-result', `Error: ${error.message}. Please run evaluation first.`);
    }
}

function displayMetrics(metrics) {
    const resultDiv = document.getElementById('metrics-result');
    
    let html = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Macro F1-Score</div>
                <div class="metric-value">${metrics.macro_f1.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">${metrics.accuracy.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">${metrics.precision.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">${metrics.recall.toFixed(4)}</div>
            </div>
        </div>
        
        <h3 style="margin-top: 30px;">Per-Class F1-Scores</h3>
        <div class="result-item">
            ${Object.entries(metrics.per_class_f1).map(([category, f1]) => `
                <p><strong>${category}:</strong> ${f1.toFixed(4)}</p>
            `).join('')}
        </div>
        
        <h3 style="margin-top: 30px;">Confusion Matrix</h3>
        <div class="confusion-matrix">
            <table>
                <thead>
                    <tr>
                        <th></th>
                        ${metrics.confusion_matrix[0].map((_, i) => `<th>Class ${i}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${metrics.confusion_matrix.map((row, i) => `
                        <tr>
                            <th>Class ${i}</th>
                            ${row.map(cell => `<td>${cell}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.classList.add('show');
}

// Load categories
async function loadCategories() {
    showLoading('categories-result');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/categories`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayCategories(data);
        populateCategorySelect(data);
    } catch (error) {
        showError('categories-result', `Error: ${error.message}`);
    }
}

function displayCategories(categories) {
    const resultDiv = document.getElementById('categories-result');
    
    let html = `<h3>Available Categories (${categories.length})</h3><div class="category-list">`;
    
    categories.forEach(cat => {
        html += `
            <div class="category-card">
                <div class="category-id">${cat.id}</div>
                <div class="category-name">${cat.name}</div>
                ${cat.description ? `<div class="category-desc">${cat.description}</div>` : ''}
            </div>
        `;
    });
    
    html += '</div>';
    resultDiv.innerHTML = html;
    resultDiv.classList.add('show');
}

function populateCategorySelect(categories) {
    const select = document.getElementById('feedback-category');
    select.innerHTML = '<option value="">Select category...</option>';
    categories.forEach(cat => {
        const option = document.createElement('option');
        option.value = cat.id;
        option.textContent = `${cat.id} - ${cat.name}`;
        select.appendChild(option);
    });
}

function showCategoryEditor() {
    alert('Category editor feature - Edit backend/config/categories.json and restart the server to update categories.');
}

// Utility functions
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    element.innerHTML = '<div class="loading">Loading...</div>';
    element.classList.add('show');
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="error">${message}</div>`;
    element.classList.add('show');
}

function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="success">${message}</div>`;
    element.classList.add('show');
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

// Load categories on page load
window.addEventListener('DOMContentLoaded', () => {
    loadCategories();
});

