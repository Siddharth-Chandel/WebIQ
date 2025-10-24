document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.querySelector('.chat-window');
    const urlInput = document.getElementById('urlInput');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const scrapeBtn = document.querySelector('.scrape-button');
    const resetBtn = document.querySelector('.reset-button');

    let ws = null;
    let sessionId = localStorage.getItem('chatbot_session') || null;

    function addMessage(msg, isUser = true) {
        const el = document.createElement('div');
        el.classList.add('message', isUser ? 'user-message' : 'bot-message');
        el.textContent = msg;
        chatWindow.appendChild(el);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function showLoading() {
        if (!document.querySelector('.loading')) {
            const loading = document.createElement('div');
            loading.classList.add('message', 'bot-message', 'loading');
            loading.textContent = '...';
            chatWindow.appendChild(loading);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    }

    function hideLoading() {
        const loading = document.querySelector('.loading');
        if (loading) loading.remove();
    }

    async function waitForChatbotReady(sessionId) {
        addMessage("Waiting for chatbot to finish initialization...", false);
        showLoading();

        while (true) {
            try {
                const statusRes = await fetch(`https://schandel08-webiq-backend.hf.space/session_status/${sessionId}`);
                const statusData = await statusRes.json();

                if (statusData.status === "ready") break;
                else console.log("⏳ Initializing...");
            } catch (err) {
                console.error("Error checking session status:", err);
            }

            await new Promise(r => setTimeout(r, 1000));
        }

        hideLoading();
        addMessage("Chatbot is ready! You can start asking questions.", false);
    }

    function initWebSocket(sessionId) {
        ws = new WebSocket(`wss://schandel08-webiq-backend.hf.space/ws/chat/${sessionId}`);

        ws.onopen = () => {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        };

        ws.onmessage = (event) => {
            hideLoading();
            try {
                const msg = JSON.parse(event.data);
                if (msg.text) addMessage(msg.text, false);
                if (msg.error) addMessage(`Error: ${msg.error}`, false);
            } catch (err) {
                console.error("Invalid JSON from server:", err);
            }
        };

        ws.onclose = () => {
            hideLoading();
            addMessage("WebSocket disconnected. Use Reset to start over.", false);
            userInput.disabled = true;
            sendBtn.disabled = true;
        };

        ws.onerror = (err) => {
            hideLoading();
            console.error(err);
            addMessage("WebSocket error. Check console.", false);
        };
    }

    async function startScrape() {
        const urls = urlInput.value.split(',').map(u => u.trim()).filter(Boolean);
        if (!urls.length) return addMessage("Enter at least one URL", false);

        scrapeBtn.disabled = true;
        urlInput.disabled = true;

        try {
            if (!sessionId) {
                const sessionRes = await fetch("https://schandel08-webiq-backend.hf.space/create_session");
                const sessionData = await sessionRes.json();
                sessionId = sessionData.session;
                localStorage.setItem('chatbot_session', sessionId);
            }

            await fetch("https://schandel08-webiq-backend.hf.space/scrape/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, urls })
            });

            await waitForChatbotReady(sessionId);
            initWebSocket(sessionId);
        } catch (err) {
            hideLoading();
            addMessage(`Failed to initialize chatbot: ${err.message}`, false);
            scrapeBtn.disabled = false;
            urlInput.disabled = false;
        }
    }

    scrapeBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        await startScrape();
    });

    sendBtn.addEventListener('click', () => {
        const question = userInput.value.trim();
        if (!question || !ws || ws.readyState !== WebSocket.OPEN) return;
        addMessage(question);
        userInput.value = '';
        ws.send(JSON.stringify({ query: question }));
        showLoading();
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendBtn.click();
    });

    resetBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.close();
        ws = null;
        chatWindow.innerHTML = `
            <div class="message bot-message">
                <p>Hello! I can answer questions using information from any URL. Please provide a URL below.</p>
            </div>`;
        urlInput.disabled = false;
        urlInput.value = '';
        scrapeBtn.disabled = false;
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;
        sessionId = null;
        localStorage.removeItem('chatbot_session');
    });

    if (sessionId) {
        addMessage("🔄 Restoring previous chatbot session...", false);
        waitForChatbotReady(sessionId).then(() => initWebSocket(sessionId));
    }
});

// Particles effect
const canvas = document.getElementById('bgParticles');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let particles = [];
for (let i=0; i<120; i++){
    particles.push({
        x: Math.random()*canvas.width,
        y: Math.random()*canvas.height,
        r: Math.random()*2+1,
        dx: Math.random()*0.5-0.25,
        dy: Math.random()*0.5-0.25
    });
}

function animateParticles(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for (let p of particles){
        ctx.beginPath();
        ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle = 'rgba(37,117,252,0.5)';
        ctx.fill();
        p.x += p.dx;
        p.y += p.dy;
        if (p.x<0||p.x>canvas.width) p.dx*=-1;
        if (p.y<0||p.y>canvas.height) p.dy*=-1;
    }
    requestAnimationFrame(animateParticles);
}
animateParticles();

// Resize canvas on window resize
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
