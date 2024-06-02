document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await progressConversation();
    });
});

async function progressConversation() {
    const userInput = document.getElementById('user-input');
    const chatbotConversation = document.getElementById('chatbot-conversation-container');
    const user_question = userInput.value;
    userInput.value = '';

    // Add human message
    const newHumanSpeechBubble = document.createElement('div');
    newHumanSpeechBubble.classList.add('speech', 'speech-human');
    newHumanSpeechBubble.textContent = user_question;
    chatbotConversation.appendChild(newHumanSpeechBubble);
    chatbotConversation.scrollTop = chatbotConversation.scrollHeight;

    try {
        // Send question to the backend
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: user_question }),
        });

        const data = await response.json();

        // Add AI message
        const newAiSpeechBubble = document.createElement('div');
        newAiSpeechBubble.classList.add('speech', 'speech-ai');
        newAiSpeechBubble.textContent = data.answer;
        chatbotConversation.appendChild(newAiSpeechBubble);
        chatbotConversation.scrollTop = chatbotConversation.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
    }
}
