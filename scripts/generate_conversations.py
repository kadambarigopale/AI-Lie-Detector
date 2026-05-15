import os
import csv
import random

# Define parameters
labels = ['truthful', 'deceptive']
relationships = ['family', 'friend']
topics = ['dinner', 'trip', 'finances', 'work', 'gossip', 'plans']
sentiments = ['positive', 'negative', 'neutral']
emotional_tones = ['sarcasm', 'aggression', 'joy', 'sadness', 'neutral', 'surprise', 'fear', 'disgust', 'affection']

# Templates for text generation based on parameters
templates = {
    'family': {
        'truthful': {
            'positive': {
                'joy': [
                    "Mom: Dinner was amazing tonight! Thank you so much.\nMe: I'm so glad you liked it!",
                    "Dad: I'm so proud of you for getting that promotion.\nMe: Thanks dad, I worked really hard for it."
                ],
                'affection': [
                    "Sister: I miss you so much, we need to catch up soon.\nMe: I miss you too! Let's do lunch this weekend.",
                    "Mom: Just wanted to say I love you and I'm thinking about you.\nMe: Love you too mom!"
                ],
            },
            'negative': {
                'aggression': [
                    "Brother: Why did you take my car without asking?!\nMe: I needed to go to the store and yours was blocking mine!",
                    "Dad: You never listen to a word I say, it's infuriating!\nMe: Because you're always yelling!"
                ],
                'sadness': [
                    "Mom: Grandma isn't doing very well today.\nMe: Oh no, I'll come visit her right away.",
                    "Sister: I feel so lonely lately since I moved.\nMe: I'm sorry to hear that. I'm always here for you."
                ],
            },
            'neutral': {
                'neutral': [
                    "Mom: What time are you coming home?\nMe: Around 6 PM.",
                    "Brother: Did you feed the dog?\nMe: Yes, about an hour ago."
                ]
            }
        },
        'deceptive': {
            'positive': {
                'joy': [
                    "Mom: How is your new job?\nMe: Oh, it's absolutely perfect! I love every second of it. (I actually hate it)",
                    "Dad: Did you like the gift?\nMe: Yes! It's the best thing I've ever received! (I plan to return it)"
                ],
                'sarcasm': [
                    "Brother: Oh sure, you're the absolute best sibling ever.\nMe: Yeah, and you're the most responsible person I know.",
                    "Mom: I just *love* cleaning up after you all day.\nMe: Well, it's your favorite hobby, right?"
                ],
            },
            'negative': {
                'aggression': [
                    "Dad: Did you break this vase?!\nMe: No! I didn't even touch it! (I knocked it over earlier)",
                    "Sister: Did you read my diary?!\nMe: Don't be ridiculous, why would I do that?! (I read it this morning)"
                ],
                'fear': [
                    "Mom: Are you sure you're okay to drive?\nMe: Yes, I'm completely fine. (My hands are shaking)",
                    "Dad: Did you spend all your savings?\nMe: No, of course not. (I have $5 left)"
                ],
            },
            'neutral': {
                'neutral': [
                    "Brother: Where were you last night?\nMe: Just studying at the library. (I was at a party)",
                    "Mom: Did you finish your homework?\nMe: Yep, all done. (Haven't started)"
                ]
            }
        }
    },
    'friend': {
         'truthful': {
            'positive': {
                'joy': [
                    "Friend: That trip was incredible!\nMe: I know, right? Best weekend ever!",
                    "Friend: Guess what? I got engaged!\nMe: OH MY GOSH! Congratulations!!"
                ],
                'surprise': [
                    "Friend: I can't believe we actually won the lottery!\nMe: This is insane! Is it really happening?!",
                    "Friend: They cancelled the exam!\nMe: Wow! Really? That's amazing news!"
                ]
            },
            'negative': {
                'aggression': [
                    "Friend: You completely ignored me at the party.\nMe: You were busy talking to everyone else!",
                    "Friend: Why are you always late?!\nMe: Traffic was terrible, get off my back!"
                ],
                'disgust': [
                    "Friend: Did you see what he was eating?\nMe: Ugh, yes, it looked absolutely revolting.",
                    "Friend: The bathroom at that club was so gross.\nMe: Tell me about it, I almost threw up."
                ]
            },
            'neutral': {
                'neutral': [
                    "Friend: Want to grab coffee?\nMe: Sure, meet you at 10.",
                    "Friend: Did you watch the game?\nMe: Yeah, it was okay."
                ]
            }
        },
        'deceptive': {
            'positive': {
                'joy': [
                    "Friend: Do you like my new haircut?\nMe: OMG it looks so amazing on you! (It looks awful)",
                    "Friend: Are you excited for the party?\nMe: So excited! Can't wait! (I really don't want to go)"
                ],
                'sarcasm': [
                    "Friend: Oh brilliant idea, let's go camping in the rain.\nMe: Yeah, because sleeping in a puddle sounds fantastic.",
                    "Friend: Your ex just walked in.\nMe: Oh perfect, just what I needed today."
                ]
            },
            'negative': {
                'aggression': [
                    "Friend: Did you tell Sarah what I said?!\nMe: No! I swear I didn't! (I told her yesterday)",
                    "Friend: You broke my phone!\nMe: It was already cracked when I got it! (I dropped it)"
                ],
                'sadness': [
                    "Friend: Are you okay? You look down.\nMe: I'm fine, just tired. (I'm really heartbroken)",
                    "Friend: I'm sorry you didn't get the job.\nMe: It's okay, I didn't want it anyway. (I'm devastated)"
                ]
            },
            'neutral': {
                'neutral': [
                    "Friend: Are you free tomorrow?\nMe: No, I have a doctor's appointment. (I just want to stay home)",
                    "Friend: How much did this cost?\nMe: Only twenty bucks. (It was hundred)"
                ]
            }
        }
    }
}

# Expand templates to create a larger pool (synthetic generation)
# In a real scenario, an LLM would generate these on the fly, but for this script,
# we will generate combinations and slight variations.

def generate_conversation(label, relationship, sentiment, tone):
    try:
        options = templates[relationship][label][sentiment][tone]
        text = random.choice(options)
        return text
    except KeyError:
        # Fallback if a specific combination isn't defined in templates
        return f"Speaker 1: Generic {tone} statement about {random.choice(topics)}.\nSpeaker 2: Generic {label} response."

def create_dataset(filename, num_records=1500):
    filepath = os.path.join(os.path.dirname(__file__), '..', filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['label', 'relationship', 'topic', 'text', 'sentiment', 'emotional_tone']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for _ in range(num_records):
            label = random.choice(labels)
            relationship = random.choice(relationships)
            topic = random.choice(topics)
            sentiment = random.choice(sentiments)
            
            # Select tone based on sentiment
            if sentiment == 'positive':
                tone = random.choice(['joy', 'affection', 'surprise', 'sarcasm'])
            elif sentiment == 'negative':
                tone = random.choice(['aggression', 'sadness', 'fear', 'disgust', 'sarcasm'])
            else:
                tone = 'neutral'
                
            text = generate_conversation(label, relationship, sentiment, tone)
            
            writer.writerow({
                'label': label,
                'relationship': relationship,
                'topic': topic,
                'text': text,
                'sentiment': sentiment,
                'emotional_tone': tone
            })
    print(f"Generated {num_records} records in {filepath}")

if __name__ == "__main__":
    # Ensure the script is run from the project root or scripts folder
    create_dataset('dataset_with_emotions.csv', num_records=2000)
