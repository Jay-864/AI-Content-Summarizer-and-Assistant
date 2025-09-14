document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileInputContainer = document.querySelector('.file-input-container');
    const uploadStatus = document.getElementById('upload-status');
    const questionForm = document.getElementById('question-form');
    const questionInput = document.getElementById('question-input');
    const submitBtn = document.getElementById('submit-btn');
    const chatMessages = document.getElementById('chat-messages');
    const statusIndicator = document.getElementById('status-indicator');
    const statusIndicatorDot = statusIndicator.querySelector('.status-indicator-dot');
    
    let isProcessing = false;
    let pollInterval;
    let userScrolledUp = false;
    let previousMessageCount = 0;
    
    // Initialize empty state
    showEmptyState();
    
    // Track user scroll behavior
    chatMessages.addEventListener('scroll', function() {
        // Check if user has scrolled up (away from the bottom)
        const isAtBottom = chatMessages.scrollHeight - chatMessages.scrollTop === chatMessages.clientHeight;
        userScrolledUp = !isAtBottom;
    });
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileInputContainer.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        fileInputContainer.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        fileInputContainer.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        fileInputContainer.classList.add('drag-over');
    }
    
    function unhighlight() {
        fileInputContainer.classList.remove('drag-over');
    }
    
    fileInputContainer.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            handleFiles(files);
        }
    }
    
    function handleFiles(files) {
        const file = files[0];
        const fileNameElement = document.querySelector('.file-input-label span');
        fileNameElement.textContent = file.name;
        
        // Show file size
        const fileSize = formatFileSize(file.size);
        document.querySelector('.file-input-label small').textContent = fileSize;
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFiles(this.files);
        }
    });
    
    // Check status periodically
    function startPollingStatus() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(checkStatus, 1000);
    }
    
    function stopPollingStatus() {
        if (pollInterval) clearInterval(pollInterval);
    }
    
    function checkStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                
                updateStatusIndicator(data.status, data.is_processing);
                
                if (data.is_processing) {
                    submitBtn.disabled = true;
                    isProcessing = true;
                } else {
                    submitBtn.disabled = false;
                    isProcessing = false;
                    
                    // If processing just finished, update messages
                    if (data.status.includes('successfully')) {
                        fetchMessages();
                    }
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                updateStatusIndicator('Connection error', false, 'error');
            });
    }
    
    function updateStatusIndicator(status, isProcessing, type = 'default') {
        statusIndicator.textContent = status;
        statusIndicator.className = 'status-indicator';
        statusIndicatorDot.className = 'status-indicator-dot';
        
        if (isProcessing) {
            statusIndicator.classList.add('status-processing');
            statusIndicatorDot.classList.add('status-processing');
        } else if (type === 'error') {
            statusIndicator.classList.add('status-error');
            statusIndicatorDot.classList.add('status-error');
        } else {
            statusIndicator.classList.add('status-ready');
            statusIndicatorDot.classList.add('status-ready');
        }
    }
    
    function fetchMessages() {
        fetch('/messages')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                
                if (data.messages.length === 0) {
                    showEmptyState();
                    previousMessageCount = 0;
                    return;
                }
                
                // Only update if messages have changed
                if (data.messages.length !== previousMessageCount) {
                    hideEmptyState();
                    
                    // Store current scroll position
                    const oldScrollHeight = chatMessages.scrollHeight;
                    const oldScrollTop = chatMessages.scrollTop;
                    
                    // Clear and rebuild messages
                    chatMessages.innerHTML = '';
                    
                    data.messages.forEach(message => {
                        addMessageToChat(message.role, message.content);
                    });
                    
                    // Adjust scroll position based on user behavior
                    if (userScrolledUp) {
                        // User has scrolled up, maintain their position relative to content
                        const newScrollHeight = chatMessages.scrollHeight;
                        const heightDifference = newScrollHeight - oldScrollHeight;
                        chatMessages.scrollTop = oldScrollTop + heightDifference;
                    } else {
                        // User is at bottom, scroll to bottom
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                    
                    previousMessageCount = data.messages.length;
                }
            })
            .catch(error => {
                console.error('Error fetching messages:', error);
            });
    }
    
    function showEmptyState() {
        if (!document.getElementById('empty-state')) {
            const emptyState = document.createElement('div');
            emptyState.id = 'empty-state';
            emptyState.className = 'empty-state';
            emptyState.innerHTML = `
                <i>💬</i>
                <p>No messages yet. Upload a file and start asking questions!</p>
            `;
            chatMessages.appendChild(emptyState);
        }
    }
    
    function hideEmptyState() {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.remove();
        }
    }
    
    function addMessageToChat(role, content) {
        hideEmptyState();
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.classList.add('message-bubble');
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        // If it's HTML content from the assistant, render it as HTML
        if (role === 'assistant' && content.includes('<')) {
            contentDiv.innerHTML = content;
        } else {
            contentDiv.textContent = content;
        }
        
        // Add timestamp
        const timeDiv = document.createElement('div');
        timeDiv.classList.add('message-time');
        timeDiv.textContent = new Date().toLocaleTimeString();
        
        bubbleDiv.appendChild(contentDiv);
        bubbleDiv.appendChild(timeDiv);
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
    }
    
    function showTypingIndicator() {
        hideEmptyState();
        
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        chatMessages.appendChild(typingDiv);
        
        // Only scroll to bottom if user hasn't scrolled up
        if (!userScrolledUp) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            showUploadStatus('Please select a file first.', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        showUploadStatus('Uploading your file...', 'processing');
        updateStatusIndicator('Uploading file...', true);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showUploadStatus(data.error, 'error');
                updateStatusIndicator('Upload failed', false, 'error');
            } else {
                showUploadStatus(data.message, 'success');
                startPollingStatus();
            }
        })
        .catch(error => {
            showUploadStatus('Upload failed: ' + error, 'error');
            updateStatusIndicator('Upload failed', false, 'error');
        });
    });
    
    function showUploadStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.className = '';
        
        if (type === 'success') {
            uploadStatus.classList.add('status-success');
        } else if (type === 'error') {
            uploadStatus.classList.add('status-error');
        } else if (type === 'processing') {
            uploadStatus.classList.add('status-processing');
        }
    }
    
    // Handle question submission
    questionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!questionInput.value.trim() || isProcessing) {
            return;
        }
        
        const question = questionInput.value.trim();
        
        // Add user question to chat immediately
        addMessageToChat('user', question);
        questionInput.value = '';
        
        // Scroll to bottom when user sends a message
        userScrolledUp = false;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Show typing indicator
        showTypingIndicator();
        
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                hideTypingIndicator();
                console.error(data.error);
                addMessageToChat('assistant', 'Error: ' + data.error);
            } else {
                startPollingStatus();
            }
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Error asking question:', error);
            addMessageToChat('assistant', 'Error: ' + error);
        });
    });
    
    // Auto-resize textarea
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Initial status check
    updateStatusIndicator('Ready', false);
    startPollingStatus();
    fetchMessages();
});