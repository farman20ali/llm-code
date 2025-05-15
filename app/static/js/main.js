document.addEventListener('DOMContentLoaded', function() {
    // Form elements
    const askForm = document.getElementById('ask-form');
    const questionInput = document.getElementById('question');
    
    const sqlForm = document.getElementById('sql-form');
    const sqlQueryInput = document.getElementById('sql-query');
    
    const jsonForm = document.getElementById('json-form');
    const jsonDataInput = document.getElementById('json-data');
    
    // Results elements
    const resultsCard = document.getElementById('results-card');
    const resultsContent = document.getElementById('results-content');
    const loadingIndicator = document.getElementById('loading');
    const rawDataToggle = document.getElementById('toggle-raw-data');
    const rawDataSection = document.getElementById('raw-data');
    const rawDataContent = document.getElementById('raw-data-content');
    
    // Toggle raw data visibility
    rawDataToggle.addEventListener('click', function() {
        rawDataSection.classList.toggle('d-none');
    });
    
    // Handle natural language to SQL form
    askForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const question = questionInput.value.trim();
        if (!question) return;
        
        showLoading();
        
        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display raw data
            rawDataContent.textContent = JSON.stringify(data, null, 2);
            
            // Display results
            const resultsHtml = `
                <h6>Generated SQL:</h6>
                <pre class="bg-light p-2 rounded">${data.sql}</pre>
                
                <h6 class="mt-4">AI Insight:</h6>
                <div class="card">
                    <div class="card-body">
                        ${data.insight}
                    </div>
                </div>
            `;
            
            hideLoading(resultsHtml);
            
        } catch (error) {
            handleError(error);
        }
    });
    
    // Handle SQL to Insights form
    sqlForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const sql = sqlQueryInput.value.trim();
        if (!sql) return;
        
        showLoading();
        
        try {
            const response = await fetch('/api/sql-insights', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sql: sql })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display raw data
            rawDataContent.textContent = JSON.stringify(data, null, 2);
            
            // Display results
            const resultsHtml = `
                <h6>SQL Query:</h6>
                <pre class="bg-light p-2 rounded">${data.sql}</pre>
                
                <h6 class="mt-4">AI Insight:</h6>
                <div class="card">
                    <div class="card-body">
                        ${data.insight}
                    </div>
                </div>
            `;
            
            hideLoading(resultsHtml);
            
        } catch (error) {
            handleError(error);
        }
    });
    
    // Handle JSON to Insights form
    jsonForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const jsonData = jsonDataInput.value.trim();
        if (!jsonData) return;
        
        showLoading();
        
        try {
            // Parse JSON data
            let parsedData;
            try {
                parsedData = JSON.parse(jsonData);
            } catch (parseError) {
                throw new Error(`Invalid JSON: ${parseError.message}`);
            }
            
            const response = await fetch('/api/json-insights', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: parsedData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display raw data
            rawDataContent.textContent = JSON.stringify(data, null, 2);
            
            // Display results
            const resultsHtml = `
                <h6 class="mt-4">AI Insight:</h6>
                <div class="card">
                    <div class="card-body">
                        ${data.insight}
                    </div>
                </div>
            `;
            
            hideLoading(resultsHtml);
            
        } catch (error) {
            handleError(error);
        }
    });
    
    // Helper functions
    function showLoading() {
        resultsCard.classList.remove('d-none');
        loadingIndicator.classList.remove('d-none');
        resultsContent.innerHTML = '';
        rawDataSection.classList.add('d-none');
    }
    
    function hideLoading(html) {
        loadingIndicator.classList.add('d-none');
        resultsContent.innerHTML = html;
    }
    
    function handleError(error) {
        console.error('Error:', error);
        loadingIndicator.classList.add('d-none');
        resultsContent.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}); 