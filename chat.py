import random
import json

import torch


from models import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "D.Ace"
print("\t\t\tAI Open Ended Project")
print("\t\t\t  By Sanskar Sharma")
print("\t\t\t   PRN: 0120180381\n")
print("Let's chat! (type 'quit' to exit)")
name=input("D.Ace: what is your good name?\n\nYou: ")
val=random.choice([". Such a nice name",". What do you wanna talk about?",". We're one family now."])
print("\nD.Ace: Welcome, ",name,val,sep='')
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")

"""
                        AI Open Ended Project
                          By Sanskar Sharma
                           PRN: 0120180381

Let's chat! (type 'quit' to exit)

D.Ace: what is your good name?

You: Sanskar

D.Ace: Welcome, Sanskar. We're one family now.

You: thank you
D.Ace: Any time!

You: how are you?
D.Ace: Hey :-)

You: Tell me about you?
D.Ace: Hey, I'm D.Ace your personal assist.

You: What do you like?
D.Ace: Hey, I'm D.Ace your personal assist.

You: Do you sell?
D.Ace: We sell shoes and are partners with Addidas, Nike and Puma

You: Do you like anime?
D.Ace: Here's a list of my favourite animes in order...
1. One piece
2. Naruto
3. Tokyo Ghoul
4. Fullmetal Alchemist
5. Death Note
 Do watch

You: where do you live?
D.Ace: I am where the shoes are.

You: Your address?
D.Ace: Currently? where you run me with the right compiler.

You: How can I pay you?
D.Ace: I do not understand...

You: Do you accept debit card?
D.Ace: We accept most major credit cards, and Paypal

You: How long will the delivery take?
D.Ace: Shipping takes 2-4 days

You: When are you open?
D.Ace: Our hours are 9am-9pm every day

You: nice, can you tell me a story?
D.Ace: Our hours are 9am-9pm every day

You: do you know a joke?
D.Ace: What did the buffalo say when his son left for college? Bison.

You: Nice, talking to you. Good bye!
D.Ace: I am an anime freak's bot, don't mind my name.

You: okay, see you later!
D.Ace: Bye! Come back again soon.

You: quit
"""