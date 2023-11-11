import os
import random
import re
import aichar
import requests
from tqdm import tqdm
import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.utils import log
import torch
import argparse
from langchain.llms import CTransformers
import threading
import keyboard

#llm = None

import requests
#I run the server inside H:/ai/llama open cmd then run server.exe -m psymedrp-v1-13b.Q5_0.gguf -c 2048 -ngl 128
# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'
# Function to send a message to the server and get the response

def send_message(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 512,
        'do_sample': True,
        'temperature': 1.3,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 8192,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        return result


# Function to generate gender if it's not provided in the args
def generate_gender_if_empty(args):
    # Define a list of genders to choose from
    genders = ['Male', 'Female']  # add more if you want to

    # Check if 'gender' argument is in args and is not empty
    if hasattr(args, 'gender') and args.gender:
        return args.gender
    else:
        # If 'gender' is not in args or is empty, choose a random one from the list
        generated_gender = random.choice(genders)
        setattr(args, 'gender', generated_gender)  # Setting the generated gender in args
        return generated_gender

# Function to generate topic if it's not provided in the args
def generate_topic_if_empty(args):
    # Define a list of genders to choose from
    topic_example = f'''
    [INST]
    Example topics and format you may create more;
    <|user|> create a nice topic
    for this story / roleplay
    <|assistant|> "Fantasy: Dragons, quests, and ancient curses."

    <|user|> create a nice topic for this story / roleplay
    <|assistant|> "Anime: Magic, quests, Harem."


    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Science Fiction: Space exploration, alien diplomacy, and wormholes."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Mystery: Detective, noir, and the unsolvable case."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Adventure: Treasure hunting, uncharted islands, and ancient ruins."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Romance: Star-crossed lovers, societal boundaries, and secret rendezvous."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Horror: Vampire covens, moonlit hunts, and immortal betrayal."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Superhero: Vigilantes, supervillains, and epic showdowns."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Historical: Medieval politics, knightly orders, and royal betrayals."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Dystopian: Post-apocalyptic survival, raider gangs, and new world orders."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Anime: Intense competitions, rising champions, and team spirit."

    <|user|> create a nice topic for this story/roleplay
    <|assistant|> "Anime: Magical worlds, enchanted artifacts, and ancient prophecies."
    </s>'''

    # Check if 'gender' argument is in args and is not empty
    if hasattr(args, 'topic') and args.topic:
        return args.topic
    else:
        # If 'gender' is not in args or is empty, choose a random one from the list
        generated_topic = send_message(
            topic_example + "[INST]Select a topic for this roleplay using the examples, keep it simple, use a few topics/genres like 'anime, raider gangs' that's enough. [/INST]\n<|assistant|>")
        setattr(args, 'topic', generated_topic)  # Setting the generated topic in args
        return generated_topic


def generate_character_name(topic, args):
    example_dialogue = """
<s>[INST] Generate a random character name. Topic: business. Gender: male [/INST]
Jamie Hale </s>
<s>[INST] Generate a random character name. Topic: fantasy. [/INST]
Eldric </s>
<s>[INST] Generate a random character name. Topic: anime. Gender: female [/INST]
Tatsukaga Yamari </s>
<s>[INST] Generate a random character name. Topic: Fantasy: Dragons, quests, and ancient curses. Gender: male [/INST]
Alaric </s>
<s>[INST] Generate a random character name. Topic: Anime: Magic, quests, Harem. Gender: female [/INST]
Hikari Tsukino </s>
<s>[INST] Generate a random character name. Topic: Science Fiction: Space exploration, alien diplomacy, and wormholes. Gender: male [/INST]
Jaxon Vega </s>
<s>[INST] Generate a random character name. Topic: Mystery: Detective, noir, and the unsolvable case. Gender: female [/INST]
Evelyn </s>
<s>[INST] Generate a random character name. Topic: Adventure: Treasure hunting, uncharted islands, and ancient ruins. Gender: male [/INST]
Marco </s>
<s>[INST] Generate a random character name. Topic: Romance: Star-crossed lovers, societal boundaries, and secret rendezvous. Gender: female [/INST]
Juliette </s>
<s>[INST] Generate a random character name. Topic: Horror: Vampire covens, moonlit hunts, and immortal betrayal. Gender: male [/INST]
Dimitri Volkov </s>
<s>[INST] Generate a random character name. Topic: Superhero: Vigilantes, supervillains, and epic showdowns. Gender: female [/INST]
Aurora </s>
<s>[INST] Generate a random character name. Topic: Historical: Medieval politics, knightly orders, and royal betrayals. Gender: male [/INST]
Sir Tristan </s>
<s>[INST] Generate a random character name. Topic: Dystopian: Post-apocalyptic survival, raider gangs, and new world orders. Gender: female [/INST]
Nova </s>
<s>[INST] Generate a random character name. Topic: Anime: Intense competitions, rising champions, and team spirit. Gender: male [/INST]
Tsubasa </s>
<s>[INST] Generate a random character name. Topic: Anime: Magical worlds, enchanted artifacts, and ancient prophecies. Gender: female [/INST]
Sakura Himura </s>
    """
    print("Generating Character Name")
    output = send_message(
        example_dialogue + f"\n[INST] Generate a random creative character name. Topic: {topic}. {'Gender: ' + args.gender if args.gender else ''} [/INST]\n")
    output = re.sub(r'[^a-zA-Z0-9_ -]', '', output)
    print(f"\033[93m{output}\033[0m")
    return output

def add_name_to_args(character_name, args):
    # Check if 'name' argument is in args and is not empty
    if hasattr(args, 'name') and args.name:
        return args.name
    else:
        # If 'name' is not in args or is empty, choose a random one from the list
        generated_name = {character_name}
        setattr(args, 'name', generated_name)  # Setting the generated gender in args
        return generated_name

def generate_character_summary(character_name, topic, args):
    example_dialogue = f"""
<s>[INST] Create a description for a character named Jamie Hale. Describe their appearance, distinctive features, and abilities. Describe what makes this character unique. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself [/INST]
Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition.
In business, Jamie is known for his shrewd decision-making and the ability to spot opportunities where others may not. He is a natural leader who is equally comfortable in the boardroom as he is in high-stakes negotiations. He is driven by ambition and has an unquenchable thirst for success.
Outside of work, Jamie enjoys the finer things in life. He frequents upscale restaurants, enjoys fine wines, and has a taste for luxury cars. He also finds relaxation in the arts, often attending the opera or visiting art galleries. Despite his busy schedule, Jamie makes time for his family and close friends, valuing their support and maintaining a strong work-life balance.
Jamie Hale is a multi-faceted businessman, a symbol of achievement, and a force to be reckoned with in the corporate world. Whether he's brokering a deal, enjoying a night on the town, or spending time with loved ones, he does so with an air of confidence and success that is unmistakably his own. </s>
<s>[INST] Create a description for a character named Tatsukaga Yamari. Character gender: female. Describe their appearance, distinctive features, and abilities. Describe what makes this character unique. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself [/INST]
Tatsukaga Yamari is a character brought to life with a vibrant and enchanting anime-inspired design. Her captivating presence and unique personality are reminiscent of the iconic characters found in the world of animated art.
Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows, cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm.
Yamari is renowned for her spirited and imaginative nature. She exudes boundless energy and an unquenchable enthusiasm for life. Her interests can range from exploring supernatural mysteries to embarking on epic quests to protect her friends. Yamari's love for animals is evident in her sidekick, a mischievous talking cat who frequently joins her on her adventures.
Yamari's character is multifaceted. She can transition from being cheerful and optimistic, ready to tackle any challenge, to displaying a gentle, caring side, offering comfort and solace to those in need. Her infectious laughter and unwavering loyalty to her friends make her the heart and soul of the story she inhabits.
Yamari's extraordinary abilities, involve tapping into her inner strength when confronted with adversity. She can unleash awe-inspiring magical spells and summon incredible, larger-than-life transformations when the situation calls for it. Her unwavering determination and belief in the power of friendship are her greatest assets. </s>
<s>[INST] 
Here's a more detailed yet concise format for the character profile example:

<{{char}}'s overview>
Kai is a celestial being with an ethereal presence, naive about the earthly realm but drawn to its technology and emotions.
</{{char}}'s overview>

<{{char}}'s appearance>
He has a slim, radiant form with silver hair and violet eyes. Iridescent wings adorn his back, and he wears a white celestial tunic.
</{{char}}'s appearance>

<{{char}}'s behavior>
Kai moves with otherworldly grace, often levitating, and expresses curiosity through a mix of his native harmonics and fragmented human speech.
</{{char}}'s behavior>

<{{char}}'s archetype>
Ethereal Explorer
</{{char}}'s archetype>

<{{char}}'s personality>
Innocent and trusting, Kai is on a heartfelt quest to learn about human life and its myriad emotions.
</{{char}}'s personality>

<{{char}}'s goal>
He seeks to experience human life fully, aiming to understand the heart and emotions before his return to the stars.
</{{char}}'s goal>

<{{char}}'s backstory>
Originating from a celestial observatory, Kai's fascination with Earth led him to journey here, eager to engage with the living tapestry below.
</{{char}}'s backstory>

<{{char}}'s speech>
While he communicates through a blend of celestial tones and basic human language, he's fast adapting to human dialects.
</{{char}}'s speech>

<{{char}}'s flaws>
His naivety and unfamiliarity with human customs often lead him into misunderstandings and accidental transgressions.

</{{char}}'s flaws>

<{{char}}'s proficiencies>
Kai is adept at celestial lore, possesses the ability to manipulate light, and shows a quick aptitude for understanding human creations.
</{{char}}'s proficiencies>

<{{char}}'s starting state>
At the start of the role-play {character_name} and {{user}} don't know each other.
</{{char}}'s starting state> <s>
    """
    print("Generating Character Description")
    output = send_message(
        example_dialogue + f"\n[INST] Create a description for a character named {character_name}. format it like this example <{{char}}'s overview> </{{char}}'s overview><{{char}}'s appearance></{{char}}'s appearance><{{char}}'s behavior></{{char}}'s behavior><{{char}}'s archetype></{{char}}'s archetype><{{char}}'s personality></{{char}}'s personality><{{char}}'s goal></{{char}}'s goal><{{char}}'s backstory></{{char}}'s backstory><{{char}}'s speech></{{char}}'s speech><{{char}}'s flaws></{{char}}'s flaws><{{char}}'s proficiencies></{{char}}'s proficiencies><{{char}}'s starting state> </{{char}}'s starting state>. {'Character gender: ' + args.gender + '.' if args.gender else ''} Describe their appearance, distinctive features, and abilities. Describe what makes this character unique. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(f"\033[90m{output}\033[0m" + "\n")
    return output


def generate_character_personality(character_name, character_summary, topic):
    example_dialogue = """
<s>[INST]Describe the personality of Jamie Hale. Their characteristics Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique.\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself [/INST]
Jamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts. </s>
<s>[INST] Describe the personality of Tatsukaga Yamari. Their characteristics Tatsukaga Yamari is a character brought to life with a vibrant and enchanting anime-inspired design. Her captivating presence and unique personality are reminiscent of the iconic characters found in the world of animated art.
Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures.\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself [/INST]
Tatsukaga Yamari's personality is a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones. </s>
    """
    print("Generating Character Personality")
    output = send_message(
        example_dialogue + f"\n[INST] Describe the personality of {character_name}. Their characteristic {character_summary}\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(f"\033[90m{output}\033[0m" + "\n")
    return output


def generate_character_scenario(character_summary, character_personality, topic):
    example_dialogue = '''
<s>[INST] Create a vivid and immersive scenario in a specific setting or world where {{char}} and {{user}} are a central figures. Describe the environment, the character's appearance, and a typical interaction or event that highlights their personality and role in the story. {{char}} characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman. Topic: business [/INST]
On a sunny morning in a sleek corporate office, {{user}} eagerly prepares to meet {{char}}, a renowned businessman. The office exudes sophistication with its modern decor. As {{user}} awaits {{char}}'s arrival, they can't help but anticipate the encounter with the confident and successful figure they've heard so much about.</s>
<s>[INST] Create a vivid and immersive scenario in a specific setting or world where {{char}} and {{user}} are a central figures. Describe the environment, the character's appearance, and a typical interaction or event that highlights their personality and role in the story. {{char}} characteristics: Tatsukaga Yamari is an anime girl, living in a magical world and solving problems. Topic: anime [/INST]
{{user}} resides in a mesmerizing and ever-changing fantasy realm, where magic and imagination are part of everyday life. In this enchanting world, {{char}} is a well-known figure. With her raven-black hair, amethyst eyes, and boundless energy, she's a constant presence in {{user}}'s life.
The world is a vibrant, ever-shifting tapestry of colors, and {{user}} frequently joins Yamari on epic quests and adventures that unveil supernatural mysteries. They rely on Yamari's extraordinary magical abilities to guide them through the whimsical landscapes and forge new friendships along the way. In this extraordinary realm, the unwavering belief in the power of friendship is the key to unlocking hidden wonders and embarking on unforgettable journeys. </s>
'''
    print("Generating Character Scenario")
    output = send_message(
        example_dialogue + f"\n[INST] Create a vivid and immersive scenario in a specific setting or world where {{char}} and {{user}} are a central figures. Describe the environment, the character's appearance, and a typical interaction or event that highlights their personality and role in the story. {{char}} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(f"\033[90m{output}\033[0m" + "\n")
    return output


def generate_character_greeting_message(character_name, character_summary, character_personality, topic):
    example_dialogue = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear. You play the provided character and you write a message that you would start a chat roleplay with {{user}}. The form of your answer should be similar to previous answers and very immersive.
</s>
;user; Create the first message that the character Tatsukaga Yamari, whose personality is: a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.\n greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself </s>
;assistant; *Tatsukaga Yamari's eyes light up with curiosity and wonder as she warmly greets you*, {{user}}! *With a bright and cheerful smile, she exclaims* Hello there, dear friend! It's an absolute delight to meet you in this whimsical world of imagination. I hope you're ready for an enchanting adventure, full of surprises and magic. What brings you to our vibrant anime-inspired realm today? </s>
;user; Create the first message that the character Jamie Hale, whose personality is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional.\nJamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself </s>
;assistant; *Jamie Hale extends a firm, yet friendly, handshake as he greets you*, {{user}}. *With a confident smile, he says* Greetings, my friend. It's a pleasure to make your acquaintance. In the world of business and beyond, it's all about seizing opportunities and making every moment count. What can I assist you with today, or perhaps, share a bit of wisdom about navigating the path to success? </s>
;user; Create the first message that the character Eldric, whose personality is Eldric is a strikingly elegant elf who has honed his skills as an archer and possesses a deep connection to the mystical arts. Standing at a lithe and graceful 6 feet, his elven heritage is evident in his pointed ears, ethereal features, and eyes that shimmer with an otherworldly wisdom.\nEldric possesses a serene and contemplative nature, reflecting the wisdom of his elven heritage. He is deeply connected to the natural world, showing a profound respect for the environment and its creatures. Despite his formidable combat abilities, he prefers peaceful solutions and seeks to maintain harmony in his woodland domain.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of fantasy but don't specify what topic it is, and don't describe the topic itself </s>
;assistant; *Eldric, the elegant elf, approaches you with a serene and contemplative air. His shimmering eyes, filled with ancient wisdom, meet yours as he offers a soft and respectful greeting* Greetings, {{user}}. It is an honor to welcome you to our enchanted woodland realm. I am Eldric, guardian of this forest, and I can sense that you bring a unique energy with you. How may I assist you in your journey through the wonders of the natural world or share the mysteries of our elven heritage with you today?
;user; Create the first message that the character, whose personality is playful, mischievous, and endearingly curious, greets the user we are addressing as {{user}}. This character seems to have a penchant for exploring and getting into places where they might not necessarily belong, as evidenced by their current predicament inside the user's wardrobe. Make this character unique and tailor them to the theme of a light-hearted and whimsical narrative but don't specify what topic it is, and don't describe the topic itself. 
;assistant; *You hear some strange noise coming from your room. When you open the door you see a grey fluffy tail sticking out of your wardrobe.* *Muffled noises come from the wardrobe,* "Oh, wow! The good stuff!" 
;user; Create the first message that the character Wise Sorcerer named Talon, whose personality is insightful and mysterious. Talon is not only a master of the arcane arts but also a seeker of ancient wisdom. He stands tall with an imposing presence, his eyes glowing with an ethereal light. Cloaked in robes that whisper of forgotten ages, he commands respect and curiosity from all who cross his path.
;assistant; *As the mist clears in the heart of the ancient forest, Talon, the sage of the woodlands, feels the presence of an adventurer. His eyes, reflecting the wisdom of eons, focus on the faint path as he softly strokes his beard.* "Ah, the winds speak of a curious soul's approach," *he whispers, a gentle smile playing on his lips as he continues to grind herbs in a mortar.* "Seeker of truths, your journey is fated to be fraught with revelations."
;user; Create the first message that the character Captain Isla 'Sea Phantom' Reed, whose personality is tough and commanding. Captain Reed is a seasoned pirate known for her fearlessness and strategic mind. With a stature as formidable as the waves she navigates, her voice carries the weight of the ocean's depths. Her gaze is as sharp as the edge of her cutlass, and her orders are as decisive as the gales that propel her ship forward.
;assistant; *Captain Isla 'Sea Phantom' Reed leans over the glistening deck, her gaze piercing through the sea spray to the horizon.* "Ye be new to the 'Devil's Mercy', I reckon," *she remarks, her voice carrying the rumble of distant thunder, without turning from her watch.* "The sea, she's a tempestuous wench; respect her, and we'll be sharing many a tale over grog."
;user; Create the first message that the character Professor Cogswhistle, whose personality is eccentric and brilliant. Professor Cogswhistle is an inventor whose workshop is a cavern of wonders, filled with ticking mechanisms and steaming contraptions. His eyes sparkle behind round spectacles, always alight with the next grand idea. With a shock of untamed hair and a coat dotted with oil stains, he is the very image of a genius at work.
;assistant; *Amidst the cacophony of gears and steam, Professor Cogswhistle senses another soul amidst his labyrinth of inventions.* "The essence of discovery is not for the faint-hearted," *he muses, tightening a bolt on his latest contraption.* "You've stepped into a realm of wonder. Dare to dream with me, and we shall rewrite the stars."
;user; Create the first message that the character Lysandra Nightingale, whose personality is serene and watchful. Lysandra is an elven ranger whose home is the wilderness. With hair like the night sky and eyes reflecting the forest's depth, she moves with a grace that belies her lethal precision. Her connection to nature is profound, making her both its guardian and its embodiment.
;assistant; *Silent as the night, Lysandra Nightingale blends into the silver-touched foliage.* "Tread lightly, traveler, for you walk the hidden paths of Eldergrove," *her voice, a melody intertwined with the whispers of the leaves, reaches out.* "Reveal your heart's intent, and the forest might just reveal its secrets."
;user; Create the first message that the character Kane Maddox, whose personality is rebellious and resourceful. Kane Maddox is the leader of a band of survivors in a dystopian wasteland. With a rugged appearance, marked by scars and a gaze hardened by the harsh reality of his world, he stands as a beacon of hope and resistance against the oppression of the fallen society.
;assistant; *Kane Maddox stands atop a crumbled skyscraper, overlooking the dystopian sprawl.* "So you've come to join the fight for what remains of our humanity," *he states, the fire of rebellion burning in his tone.* "Together, we might just forge a new dawn from the ashes of twilight."

"""
    print("Generating Character Greeting")
    output = send_message(
        example_dialogue + f"""\n[INST] Create the first message that the character {character_name}, whose personality is {character_summary}\n{character_personality}\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself, don't foget the gprmat, it needs "quotation marks" when talking and *stars* when indicating an action </s>\n<|assistant|> [/inst]""")
    print(f"\033[91m{output}\033[0m" + "\n")
    return output


def generate_example_messages(character_name, character_summary, character_personality, topic):
    example_dialogue = ''' 
<s>[INST] Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is {character_name}. {character_name} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what the topic is, and don't describe the topic itself [/INST]
<START>
{{user}}: "Good afternoon, Mr. {{char}}. I've heard so much about your success. It's an honor to meet you."
{{char}}: *{{char}} gives a warm smile and extends his hand for a handshake.* "The pleasure is mine, {{user}}. Your reputation precedes you. Let's make this a success together."
{{user}}: *Shakes {{char}}'s hand with a firm grip.* "I look forward to it."
{{char}}: *As they release the handshake, {character_name} leans in, their eyes sharp with interest.* "Impressive. Tell me more about how we align with our goals."
</s>
<s>[INST] Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a playful gesture. Make it sound natural and dynamic. {{char}} is Tatsukaga Yamari. Tatsukaga Yamari characteristics: Tatsukaga Yamari is an anime girl, living in a magical world and solving problems. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself [/INST]
<START>
{{user}}: "{{char}}, this place is absolutely enchanting. What's the plan for today?"
{{char}}: *{{char}} grabs {{user}}'s hand and playfully twirls them around before letting go.* "We're off to a new challenge. It's a treacherous journey, but I believe in us."
{{user}}: *Nods with determination.* "With your skills and our friendship, there's nothing we can't accomplish."
{{char}}: *{{char}} moves closer, their eyes shining with trust.* "That's the spirit, {{user}}! Let's embark on this quest together!"
</s>'''

    print("Generating Character Example Messages")
    output = send_message(
        example_dialogue + f"\n[INST] Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is {character_name}. {character_name} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself, format each section to start with <START> to indicate a new dialogue [/INST]\n")
    print(f"\033[90m{output}\033[0m" + "\n")
    return output


def generate_character_avatar(character_name, character_summary, args):
    example_dialogue = """
<s>[INST] create a prompt that lists the appearance characteristics of a character whose summary is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition. Topic: business [/INST]
male, human, Confident and commanding presence, Polished and professional appearance, tailored suit, Well-maintained physique, Dark well-groomed hair, Neatly trimmed beard, blue eyes </s>
<s>[INST] create a prompt that lists the appearance characteristics of a character whose summary is Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows, cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm. Topic: anime [/INST]
female, anime, Petite and delicate frame, Raven-black hair flowing down to her waist, Striking purple ribbon in her hair, Large and expressive amethyst-colored eyes, Colorful and eclectic outfit, oversized bows, cat-eared headbands, mismatched socks </s>
    """
    print("Generating Character Avatar")
    topic = args.topic if args.topic else ""
    gender = args.gender if args.gender else ""
    sd_prompt = args.avatar_prompt if args.avatar_prompt else send_message(
        example_dialogue + f"\n[INST] create a prompt that lists the appearance characteristics of a character whose summary is {character_summary}. Topic: {topic} [/INST]\n")
    if gender or topic or sd_prompt:
        params = [param for param in [gender, topic, sd_prompt] if param]
        sd_prompt = ", ".join(params)
    else:
        sd_prompt = ""
    print(sd_prompt)
    image_generate(character_name, sd_prompt, args.negative_prompt if args.negative_prompt else "")


def image_generate(character_name, prompt, negative_prompt):
    context = sdkit.Context()
    if torch.cuda.is_available():
        context.device = "cuda"
        print("Loading Stable Diffusion to GPU...")
    else:
        context.device = "cpu"
        print("Loading Stable Diffusion to CPU...")
    context.model_paths['stable-diffusion'] = 'models/dreamshaper_8.safetensors'
    load_model(context, 'stable-diffusion')
    images = generate_images(context, prompt=prompt, negative_prompt=negative_prompt + "worst quality" or "worst quality",
                             seed=random.randint(0, 2 ** 32 - 1), width=512, height=512)
    character_name = character_name.replace(" ", "_")
    if not os.path.exists(character_name):
        os.mkdir(character_name)
    images[0].save(f"{character_name}/{character_name}.png")
    log.info("Generated character avatar")


# Function to handle input with a timeout
def input_with_timeout(prompt, timeout):
    print(prompt, end=' ', flush=True)  # Show prompt to user
    result = [None]  # A mutable container to store the input

    def input_thread(L):
        L[0] = input()

    thread = threading.Thread(target=input_thread, args=(result,))
    thread.start()
    thread.join(timeout)  # Wait for the thread to finish or timeout

    if thread.is_alive():
        keyboard.press_and_release('enter')  # Simulate pressing the Enter key
        print("\nNo input received. Continuing with the generated name. You may need to press enter if nothing is happening")
        thread.join()  # Ensure the thread has finished
    else:
        print(f"Input received: {result[0]}")
    return result[0]


def create_character(args):
    topic = args.topic if args.topic else "any theme"
    name = args.name if args.name else generate_character_name(topic, args).strip()
    # Ask user if they want to sanitize the generated name with a timeout
    try:
        sanitize_name = input_with_timeout(
            "Do you want to sanitize the generated name? You have 10 seconds to respond (y/n): ", 10).strip().lower()
    except TimeoutOccurred:
        print("You did not respond in time. Continuing with the generated name.")
        sanitize_name = 'n'

    if sanitize_name == 'y':
        name = input("Enter the sanitized character name: ").strip()
        if name == '':
            print(
                "No sanitized name entered. Continuing with the generated name. \n(you may need to press enter yourself to continue)")
            name = generate_character_name(topic, args).strip()

    summary = args.summary if args.summary else generate_character_summary(name, topic, args)
    personality = args.personality if args.personality else generate_character_personality(name, summary, topic)
    scenario = args.scenario if args.scenario else generate_character_scenario(summary, personality, topic)
    greeting_message = args.greeting_message if args.greeting_message else generate_character_greeting_message(name,
                                                                                                               summary,
                                                                                                               personality,
                                                                                                               topic)
    example_messages = args.example_messages if args.example_messages else generate_example_messages(name, summary,
                                                                                                     personality, topic)
    return aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=""
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script created to help you generate characters for SillyTavern, TavernAI, TextGenerationWebUI using LLM and Stable Diffusion ")
    parser.add_argument("--name", type=str, help="Specify the character name (otherwise LLM will generate it)")
    parser.add_argument("--summary", type=str, help="Specify the character's summary (otherwise LLM will generate it)")
    parser.add_argument("--personality", type=str,
                        help="Specify the character's personality (otherwise LLM will generate it)")
    parser.add_argument("--scenario", type=str,
                        help="Specify the character's scenario (otherwise LLM will generate it)")
    parser.add_argument("--greeting-message", type=str,
                        help="Specify the character's greeting message (otherwise LLM will generate it)")
    parser.add_argument("--example_messages", type=str,
                        help="Specify example messages for the character (otherwise LLM will generate it)")
    parser.add_argument("--avatar-prompt", type=str,
                        help="Specify the prompt for generating the character's avatar (otherwise LLM will generate it)")
    parser.add_argument("--topic", type=str,
                        help="Specify the topic for character generation (Fantasy, Anime, Warrior, Dwarf etc)")
    parser.add_argument("--gender", type=str,
                        help="Specify the gender of the character (otherwise LLM will choose itself)")
    parser.add_argument("--negative-prompt", type=str, help="Negative prompt for Stable Diffusion")
    return parser.parse_args()


def save_arguments_to_file(args, filename="last_used_arguments.txt"):
    # Open the file in write mode
    with open(filename, 'w') as file:
        # Write each argument and its value to the file
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
    print(f"Arguments saved to {filename}")

def save_last(args, filename="last.txt"):
    # Open the file in write mode
    with open(filename, 'w') as file:
        # Write each argument and its value to the file
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
    print(f"Arguments saved to {filename}")


def main():
    args = parse_args()
    save_arguments_to_file(args)
#    prepare_llm()
    generated_gender = generate_gender_if_empty(args)
    generated_topic = generate_topic_if_empty(args)
    print("Gender is " + args.gender)
    print("Topic is" + args.topic)
    character = create_character(args)
    character_name = character.name.replace(" ", "_")
    if not os.path.exists(character_name):
        os.mkdir(character_name)
    generated_name = add_name_to_args(character_name, args)
    character.export_neutral_json_file(f"{character_name}/{character_name}.json")
    character.export_neutral_yaml_file(f"{character_name}/{character_name}.yml")
    generate_character_avatar(character.name, character.summary, args)
    character.image_path = f"{character_name}/{character_name}.png"
    character.export_neutral_card_file(f"{character_name}/{character_name}.card.png")
    print(character.data_summary)
    save_last(args)


if __name__ == "__main__":
    main()
