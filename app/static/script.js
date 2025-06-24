// Water Meter Scanner - Basic JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('Water Meter Scanner loaded');
    
    // Upload form handling
    const uploadForm = document.getElementById('uploadForm');
    const uploadResult = document.getElementById('uploadResult');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('image');
            const cityInput = document.getElementById('city');
            const streetInput = document.getElementById('street');
            const numberInput = document.getElementById('number');
            
            // Validate inputs
            if (!fileInput.files[0]) {
                showResult('Please select an image file', 'error');
                return;
            }
            
            if (!cityInput.value.trim() || !streetInput.value.trim() || !numberInput.value.trim()) {
                showResult('Please fill in all address fields', 'error');
                return;
            }
            
            // Prepare form data
            formData.append('file', fileInput.files[0]);
            formData.append('city', cityInput.value.trim());
            formData.append('street_name', streetInput.value.trim());
            formData.append('street_number', numberInput.value.trim());
            
            // Show loading
            showResult('Processing image... This may take a few moments.', 'loading');
            
            try {
                const response = await fetch('/upload-meter', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok && result.success) {
                    showResult(`
                        <h3>✅ Successfully Extracted Reading!</h3>
                        <p><strong>Address:</strong> ${result.address}</p>
                        <p><strong>Meter Value:</strong> ${result.meter_value}</p>
                        <p><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</p>
                        <p><strong>Reading ID:</strong> ${result.reading_id}</p>
                        ${result.notes ? `<p><strong>Notes:</strong> ${result.notes}</p>` : ''}
                    `, 'success');
                    
                    // Reset form
                    uploadForm.reset();
                } else {
                    showResult(`❌ Error: ${result.detail || result.error || 'Unknown error'}`, 'error');
                }
                
            } catch (error) {
                showResult(`❌ Network error: ${error.message}`, 'error');
            }
        });
    }
    
    function showResult(message, type) {
        if (uploadResult) {
            uploadResult.innerHTML = message;
            uploadResult.className = `result-section ${type}`;
        }
    }
    
    // Chat input handling (still disabled)
    const chatInput = document.querySelector('.chat-input input');
    const chatButton = document.querySelector('.chat-input button');
    
    if (chatInput && chatButton) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !chatInput.disabled) {
                handleChatSubmit();
            }
        });
        
        chatButton.addEventListener('click', function() {
            if (!chatButton.disabled) {
                handleChatSubmit();
            }
        });
    }
    
    function handleChatSubmit() {
        alert('Chat functionality not yet implemented');
    }
});