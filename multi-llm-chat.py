import os
import json
from typing import List, Dict
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import VertexAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

class LLMChatApp:
    def __init__(self):
        self.api_providers = {
            'openai': ChatOpenAI(temperature=0.7),
            'anthropic': ChatAnthropic(temperature=0.7),
            'gemini': VertexAI(model_name="gemini-pro")
        }
        self.current_provider = 'openai'
        self.threads: Dict[str, List[Dict]] = {}
        self.current_thread = 'default'
        self.memories = {
            'openai': ConversationBufferMemory(return_messages=True),
            'anthropic': ConversationBufferMemory(return_messages=True),
            'gemini': ConversationBufferMemory(return_messages=True)
        }
        self.load_chat_history()

    def load_chat_history(self):
        if os.path.exists('chat_history.json'):
            with open('chat_history.json', 'r') as f:
                self.threads = json.load(f)
            # Restore memory for the current thread
            if self.current_thread in self.threads:
                for message in self.threads[self.current_thread]:
                    if message['role'] == 'user':
                        for memory in self.memories.values():
                            memory.chat_memory.add_user_message(message['content'])
                    elif message['role'] == 'assistant':
                        for memory in self.memories.values():
                            memory.chat_memory.add_ai_message(message['content'])
        if self.current_thread not in self.threads:
            self.threads[self.current_thread] = []

    def save_chat_history(self):
        with open('chat_history.json', 'w') as f:
            json.dump(self.threads, f)

    def switch_provider(self, provider: str):
        if provider in self.api_providers:
            old_provider = self.current_provider
            self.current_provider = provider
            self.adapt_context_for_new_provider(old_provider, provider)
            print(f"Switched to {provider} API")
        else:
            print(f"Provider {provider} not available")

    def adapt_context_for_new_provider(self, old_provider: str, new_provider: str):
        # Summarize the context from the old provider and add it to the new provider's memory
        old_memory = self.memories[old_provider]
        new_memory = self.memories[new_provider]
        
        summary = f"This conversation was previously handled by the {old_provider} API. "
        summary += "Here's a brief summary of the conversation so far: "
        summary += " ".join([m.content for m in old_memory.chat_memory.messages if isinstance(m, (HumanMessage, AIMessage))][-5:])
        
        new_memory.chat_memory.add_system_message(summary)

    def switch_thread(self, thread_name: str):
        self.current_thread = thread_name
        if thread_name not in self.threads:
            self.threads[thread_name] = []
        # Clear current memories and load memory for the new thread
        for memory in self.memories.values():
            memory.clear()
        for message in self.threads[thread_name]:
            if message['role'] == 'user':
                for memory in self.memories.values():
                    memory.chat_memory.add_user_message(message['content'])
            elif message['role'] == 'assistant':
                for memory in self.memories.values():
                    memory.chat_memory.add_ai_message(message['content'])
        print(f"Switched to thread: {thread_name}")

    def get_prompt_for_provider(self, provider: str):
        if provider == 'openai':
            return ChatPromptTemplate.from_messages([
                SystemMessage(content="You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully."),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{input}")
            ])
        elif provider == 'anthropic':
            return ChatPromptTemplate.from_messages([
                SystemMessage(content="You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest."),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{input}")
            ])
        elif provider == 'gemini':
            return ChatPromptTemplate.from_messages([
                SystemMessage(content="You are Gemini, an AI model developed by Google. You're helpful, creative, and able to understand and respond to a wide variety of tasks."),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{input}")
            ])

    def chat(self, user_input: str):
        self.threads[self.current_thread].append({"role": "user", "content": user_input})
        
        conversation = ConversationChain(
            llm=self.api_providers[self.current_provider],
            memory=self.memories[self.current_provider],
            prompt=self.get_prompt_for_provider(self.current_provider),
            verbose=True
        )

        response = conversation.predict(input=user_input)
        self.threads[self.current_thread].append({"role": "assistant", "content": response})
        self.save_chat_history()
        return response

def main():
    app = LLMChatApp()
    session = PromptSession(history=FileHistory('.chat_history'))

    while True:
        try:
            user_input = session.prompt(
                "You: ",
                auto_suggest=AutoSuggestFromHistory()
            )

            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.startswith('/switch '):
                app.switch_provider(user_input.split()[1])
            elif user_input.startswith('/thread '):
                app.switch_thread(user_input.split()[1])
            else:
                response = app.chat(user_input)
                print(f"Assistant: {response}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("Goodbye!")

if __name__ == "__main__":
    main()