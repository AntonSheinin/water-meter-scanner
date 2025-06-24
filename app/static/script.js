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
    
    // Chat event listeners
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');
    
    if (chatInput && chatSend) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && chatInput.value.trim()) {
                sendChatMessage();
            }
        });
        
        chatSend.addEventListener('click', function() {
            if (chatInput.value.trim()) {
                sendChatMessage();
            }
        });
    }
});

// GLOBAL CHAT FUNCTIONS (moved outside DOMContentLoaded)
async function sendChatMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    console.log('Sending message:', message);
    
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    
    // Clear input and show loading
    chatInput.value = '';
    addChatMessage('Thinking...', 'assistant', true);
    
    try {
        console.log('Making API call...');
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        console.log('Response status:', response.status);
        const result = await response.json();
        console.log('Response data:', result);
        
        // Remove loading message
        removeLastMessage();
        
        if (response.ok) {
            addChatMessage(result.response, 'assistant');
        } else {
            addChatMessage(`Error: ${result.detail || 'Unknown error'}`, 'error');
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeLastMessage();
        addChatMessage(`Network error: ${error.message}`, 'error');
    }
}

function addChatMessage(message, sender, isLoading = false) {
    console.log('Adding message:', message, 'sender:', sender);
    
    const chatHistory = document.getElementById('chatHistory');
    if (!chatHistory) {
        console.error('chatHistory element not found!');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender} ${isLoading ? 'loading' : ''}`;
    messageDiv.innerHTML = `<div class="message-content">${message}</div>`;
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function removeLastMessage() {
    const chatHistory = document.getElementById('chatHistory');
    if (!chatHistory) return;
    const lastMessage = chatHistory.querySelector('.loading');
    if (lastMessage) {
        lastMessage.remove();
    }
}