document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.querySelector('.chat-window');
    const urlInput = document.getElementById('urlInput');
    const fullWebsiteToggle = document.getElementById('fullWebsiteToggle');
    const scrapeBtn = document.getElementById('scrapeBtn');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    
    // Global variable to hold the WebSocket connection
    let ws;

    // Function to add a message to the chat window
    function addMessage(message, isUser = true) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user-message' : 'bot-message');
        messageElement.innerHTML = `<p>${message}</p>`;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Function to add a loading indicator
    let loadingIndicator = null;
    function showLoadingIndicator() {
        if (!loadingIndicator) {
            loadingIndicator = document.createElement('div');
            loadingIndicator.classList.add('message', 'bot-message', 'loading');
            loadingIndicator.innerHTML = '<p>...</p>';
            chatWindow.appendChild(loadingIndicator);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    }

    function hideLoadingIndicator() {
        if (loadingIndicator) {
            loadingIndicator.remove();
            loadingIndicator = null;
        }
    }

    // Function to handle scraping and sending URLs to the backend
    scrapeBtn.addEventListener('click', async () => {
        const urlValue = urlInput.value.trim();
        if (!urlValue) {
            addMessage("Please enter at least one URL to scrape.", false);
            return;
        }

        const scrapeFullWebsite = fullWebsiteToggle.checked;
        
        let urls;
        if (scrapeFullWebsite) {
            urls = urlValue.split(',').map(url => url.trim()).filter(url => url);
            if (urls.length > 1) {
                addMessage("To scrape a full website, please provide only one URL.", false);
                return;
            }
        } else {
            urls = urlValue.split(',').map(url => url.trim()).filter(url => url);
        }

        // Disable input while communicating with the backend
        urlInput.disabled = true;
        scrapeBtn.disabled = true;
        fullWebsiteToggle.disabled = true;
        addMessage("Requesting chatbot initialization...", false);
        showLoadingIndicator();

        try {
            // Step 1: Send a fetch request to start the background task on the server
            const response = await fetch("http://127.0.0.1:8000/scrape/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    urls: urls,
                    llm_model: "default",
                    embedding_model: "default",
                    api_key: "default"
                })
            });

            const data = await response.json();
            hideLoadingIndicator();
            addMessage(data.message, false);
            const sessionId = data.session_id;
            console.log("Session ID:", sessionId);

            if (response.ok) {
                // Step 2: If the initial request was successful, open the WebSocket connection
                addMessage("Opening WebSocket for real-time communication...", false);
                showLoadingIndicator();
                
                ws = new WebSocket(`ws://127.0.0.1:8000/ws/chat/${sessionId}`);

                ws.onopen = () => {
                    hideLoadingIndicator();
                    addMessage("WebSocket connected. You can now ask questions!", false);
                    userInput.disabled = false;
                    sendBtn.disabled = false;
                    userInput.focus();
                };

                ws.onmessage = (event) => {
                    hideLoadingIndicator();
                    const data = JSON.parse(event.data);
                    if (data.error) {
                        addMessage(`Error: ${data.error}`, false);
                    } else if (data.text) {
                        addMessage(data.text, false);
                    }
                };

                ws.onclose = (event) => {
                    hideLoadingIndicator();
                    addMessage("Connection closed. Please refresh to start a new session.", false);
                    console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}`);
                    userInput.disabled = true;
                    sendBtn.disabled = true;
                };

                ws.onerror = (error) => {
                    hideLoadingIndicator();
                    addMessage("WebSocket error occurred. See console for details.", false);
                    console.error("WebSocket Error:", error);
                };
            } else {
                // Re-enable inputs on fetch error
                urlInput.disabled = false;
                scrapeBtn.disabled = false;
                fullWebsiteToggle.disabled = false;
            }
        } catch (error) {
            hideLoadingIndicator();
            addMessage(`Failed to connect to the server. Error: ${error.message}`, false);
            console.error("Fetch Error:", error);
            // Re-enable inputs on fetch error
            urlInput.disabled = false;
            scrapeBtn.disabled = false;
            fullWebsiteToggle.disabled = false;
        }
    });

    // Function to handle user questions
    sendBtn.addEventListener('click', () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessage("Not connected to the chatbot. Please scrape URLs first.", false);
            return;
        }

        const question = userInput.value.trim();
        if (!question) return;

        addMessage(question);
        userInput.value = '';
        showLoadingIndicator();

        // Send the new query over the existing WebSocket connection
        ws.send(JSON.stringify({
            query: question,
        }));
    });

    // Enable sending messages with the Enter key
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });

    // Initially disable the user question input until URLs are scraped
    userInput.disabled = true;
    sendBtn.disabled = true;
});