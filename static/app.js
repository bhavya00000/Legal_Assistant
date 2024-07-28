class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            readAloudButton: document.querySelector('.read__button') // Add a button for text-to-speech
        };

        this.state = false;
        this.messages = [];
        this.recognition = new webkitSpeechRecognition() || SpeechRecognition();
        this.recognition.lang = 'en-US'; // Set the language to English by default
        this.isListening = false;
    }

    display() {
        const { openButton, chatBox, sendButton, readAloudButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })

        // Add event listener for text-to-speech
        readAloudButton.addEventListener('click', () => this.readAloud(chatBox));
        const speechToTextButton = document.querySelector('#speech-to-text-button');
        speechToTextButton.addEventListener('click', () => this.toggleSpeechRecognition());
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // Show or hide the box
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    toggleSpeechRecognition() {
        if (this.isListening) {
            this.recognition.stop();
            this.isListening = false;
        } else {
            this.recognition.start();
            this.isListening = true;
            this.recognition.onresult = (event) => {
                const message = event.results[0][0].transcript;
                this.sendMessage(message);
            };
        }
    }

    sendMessage(message) {
        // Add the message to your chat interface, similar to how you handle text messages
        const msg1 = { name: 'User', message };
        this.messages.push(msg1);

        // Send the message to your server (modify this as needed)
        fetch('/api/chat', {
            method: 'POST',
            body: JSON.stringify({ message }),
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then((response) => response.json())
            .then((data) => {
                const msg2 = { name: 'Sam', message: data.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox);
            })
            .catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
            });
    }


    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';

        })
        .catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }

    // Function to read the chat messages aloud
    readAloud(chatbox) {
        // Find the last chatbot response in the messages array
        const lastChatbotResponse = this.messages.slice().reverse().find(msg => msg.name === "Sam");
    
        if (lastChatbotResponse) {
            // Use the Web Speech API to read the last chatbot response aloud
            const utterance = new SpeechSynthesisUtterance(lastChatbotResponse.message);
            window.speechSynthesis.speak(utterance);
        }
    }
    
}

const chatbox = new Chatbox();
chatbox.display();