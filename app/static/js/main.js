document.addEventListener('DOMContentLoaded', function() {
    // Form elements
    const askForm = document.getElementById('ask-form');
    const questionInput = document.getElementById('question');
    
    const langchainForm = document.getElementById('langchain-form');
    const langchainQuestionInput = document.getElementById('langchain-question');
    
    const sqlForm = document.getElementById('sql-form');
    const sqlQueryInput = document.getElementById('sql-query');
    
    const jsonForm = document.getElementById('json-form');
    const jsonDataInput = document.getElementById('json-data');
    
    const queryForm = document.getElementById('queryForm');
    const queryQuestionInput = document.getElementById('query-question');
    
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
    
    // Handle LangChain SQL form
    langchainForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const question = langchainQuestionInput.value.trim();
        if (!question) return;
        
        showLoading();
        
        try {
            const response = await fetch('/api/sql/generate-sql', {
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
            let resultsHtml = '';
            
            if (data.status === 'error') {
                resultsHtml = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> ${data.error}
                    </div>
                `;
            } else {
                resultsHtml = `
                    <h6>Generated SQL:</h6>
                    <pre class="bg-light p-2 rounded">${data.sql || 'No SQL generated'}</pre>
                `;
            }
            
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
    
    // Handle Query and Insights form
    if (queryForm) {
        queryForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            console.log('Query form submitted'); // Debug log
            
            const question = queryQuestionInput.value.trim();
            if (!question) return;
            
            // Show loading
            document.querySelector('.loading').style.display = 'block';
            document.querySelector('.result-section').style.display = 'none';
            
            try {
                console.log('Sending request to /query/generate-query-and-insights'); // Debug log
                const response = await fetch('/query/generate-query-and-insights', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });
                
                console.log('Response received:', response.status); // Debug log
                
                // Check if response is JSON
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    throw new Error('Server returned non-JSON response');
                }
                
                const data = await response.json();
                console.log('Data received:', data); // Debug log
                
                if (!response.ok) {
                    throw new Error(data.error || 'Server error');
                }
                
                if (data.status === 'error') {
                    throw new Error(data.error || 'An error occurred');
                }
                
                // Display insights
                document.getElementById('insights').textContent = data.insights;
                
                // Display SQL
                document.getElementById('sql').textContent = data.sql;
                
                // Create and display results table
                const table = document.createElement('table');
                table.className = 'table table-striped table-bordered';
                
                // Add header
                const thead = document.createElement('thead');
                const headerRow = document.createElement('tr');
                data.results.columns.forEach(col => {
                    const th = document.createElement('th');
                    th.textContent = col;
                    headerRow.appendChild(th);
                });
                thead.appendChild(headerRow);
                table.appendChild(thead);
                
                // Add body
                const tbody = document.createElement('tbody');
                data.results.rows.forEach(row => {
                    const tr = document.createElement('tr');
                    row.forEach(cell => {
                        const td = document.createElement('td');
                        td.textContent = cell;
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);
                
                // Clear and add table
                const resultsTable = document.getElementById('resultsTable');
                resultsTable.innerHTML = '';
                resultsTable.appendChild(table);
                
                // Show results
                document.querySelector('.result-section').style.display = 'block';
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            } finally {
                // Hide loading
                document.querySelector('.loading').style.display = 'none';
            }
        });
    } else {
        console.error('Query form not found!'); // Debug log
    }
    
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