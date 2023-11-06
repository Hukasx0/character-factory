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

llm = None


def prepare_llm():
    global llm
    folder_path = 'models'
    model_url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    llm_model_name = os.path.join(folder_path, os.path.basename(model_url))
    if not os.path.exists(llm_model_name):
        try:
            print(f'Downloading LLM model from: {model_url}')
            with requests.get(model_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
                with open(llm_model_name, 'wb') as out_file:
                    for data in response.iter_content(chunk_size=block_size):
                        out_file.write(data)
                        progress_bar.update(len(data))
                progress_bar.close()
            print(f'Model downloaded and saved to: {llm_model_name}')
        except Exception as e:
            print(f'Error while downloading LLM model: {str(e)}')
    sd_model_url = 'https://civitai.com/api/download/models/128713'
    sd_model_name = os.path.join(folder_path, 'dreamshaper_8.safetensors')
    if not os.path.exists(sd_model_name):
        try:
            print(f'Downloading Stable Diffusion model from: {sd_model_url}')
            with requests.get(sd_model_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
                with open(sd_model_name, 'wb') as out_file:
                    for data in response.iter_content(chunk_size=block_size):
                        out_file.write(data)
                        progress_bar.update(len(data))
                progress_bar.close()
            print(f'Model downloaded and saved to: {sd_model_name}')
        except Exception as e:
            print(f'Error while downloading Stable Diffusion model: {str(e)}')
    gpu_layers = 0
    if torch.cuda.is_available():
        gpu_layers = 110
        print("Loading LLM to GPU...")
    else:
        print("Loading LLM to CPU...")
    llm = CTransformers(
        model="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=gpu_layers,
        config={'max_new_tokens': 512,
                'repetition_penalty': 1.1,
                'top_k': 40,
                'top_p': 0.95,
                'temperature': 0.8,
                'context_length': 4096,
                'gpu_layers': gpu_layers,
                'stop': ["/s", "</s>", "<s>", "[INST]", "[/INST]", "<|im_end|>"]}
    )


def generate_character_name(topic, args):
    example_dialogue = """
<s>[INST] Generate a random character name. Topic: business. Gender: male [/INST]
Jamie Hale</s>
<s>[INST] Generate a random character name. Topic: fantasy. [/INST]
Eldric</s>
<s>[INST] Generate a random character name. Topic: anime. Gender: female [/INST]
Tatsukaga Yamari</s>
    """
    print("Generating Character Name")
    output = llm(
        example_dialogue + f"\n[INST] Generate a random character name. Topic: {topic}. {'Gender: ' + args.gender if args.gender else ''} [/INST]\n")
    output = re.sub(r'[^a-zA-Z0-9_ -]', '', output)
    print(output)
    return output


def generate_character_summary(character_name, topic, args):
    example_dialogue = """
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
    """
    print("Generating Character Summary")
    output = llm(
        example_dialogue + f"\n[INST] Create a description for a character named {character_name}. {'Character gender: ' + args.gender + '.' if args.gender else ''} Describe their appearance, distinctive features, and abilities. Describe what makes this character unique. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(output + "\n")
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
    output = llm(
        example_dialogue + f"\n[INST] Describe the personality of {character_name}. Their characteristic {character_summary}\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(output + "\n")
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
    output = llm(
        example_dialogue + f"\n[INST] Create a vivid and immersive scenario in a specific setting or world where {{char}} and {{user}} are a central figures. Describe the environment, the character's appearance, and a typical interaction or event that highlights their personality and role in the story. {{char}} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(output + "\n")
    return output


def generate_character_greeting_message(character_name, character_summary, character_personality, topic):
    # Generate a dynamic greeting based on character details and topic
    example_dialogue = f"""
    [INST] Create a unique and immersive greeting message to greet {{user}}, speaking and acting as {character_name}.
    This character is known for being {character_summary} and has a personality that is {character_personality}.
    The greeting should creatively link to the topic of {topic} and reflect the character's unique traits and personality.

    The message should place {{user}} and {character_name} in a specific, vividly described scenario that relates to the topic. 
    Include sensory details, the character's actions, thoughts, and dialogue, crafting a brief but engaging narrative encounter. 
    Avoid generic statements; aim for a personalized and memorable interaction that could only happen between {{user}} and {character_name}.
    Do not generate responses for {{user}}, only for {character_name}.

    Avoid simply replicating the structure and content of the example provided. Instead, use the example as a springboard for crafting an entirely original and context-specific greeting.

    Bad Examples:
    1. *As {{user}} enters the room, {character_name} looks up from their work on a device related to {topic}, and says: "Ah, {{user}}, I knew you'd come. Let's discuss {topic}."*
       This is too similar to the provided example and lacks originality and specific sensory and environmental details.

    2. *{{user}} walks in, and without missing a beat, {character_name} gestures broadly to a wall of related {topic} paraphernalia, proclaiming, "Welcome, {{user}}! You're just in time to talk about {topic}."*
       This example is overly direct and fails to incorporate the character's {character_personality} personality or provide a rich, sensory experience.

    Good Examples:
    1. *The aroma of ancient tomes and the whisper of turning pages envelop {{user}} as they step across the threshold into {character_name}'s sanctum. With a {character_personality} glint in their eye, {character_name} gently closes a hefty, leather-bound grimoire that hums with the essence of {topic}.*

       {character_name}: "The threads of fate are curious indeed, {{user}}. To what do I owe the pleasure of this rare visit? Or is it the enigma of {topic} that beckons you to my library's secrets?"

    2. *A cacophony of mechanical harmonies greets {{user}} as they duck into {character_name}'s wind-powered workshop. Amidst the symphony of spinning gears and fluttering propellers, {character_name}, true to their {character_personality} nature, is engrossed in a delicate dance of creation.*

       {character_name}: "Ah, {{user}}, you arrive like a gust on a calm day, unexpected yet invigorating! I have a contraption here that's just brimming with questions about {topic}. Shall we let curiosity chart our course tonight?"

    3. *Chill air and the faint scent of ozone cling to {{user}} as they step into the stark lab where {character_name}, with their {character_personality} presence, manipulates streaks of artificial lightning. The arcs pulse in time with {topic}, seemingly alive.*

       {character_name}: "Your timing is as impeccable as ever, {{user}}. It's not every day my experiments resonate with the energy of {topic}. But I suspect you're here for more than just observation. What's the real reason behind your visit?"

    These examples provide unique settings, incorporate the character's {character_personality} nature, and introduce the topic of {topic} in a subtle and natural way, without directly copying the original example.

    [END INST]
    """
    print("Generating Character Greeting Message")
    # Replace 'llm(prompt)' with the actual AI model call
    output = llm(
        example_dialogue + f" [INST] Initiate an engaging and natural-sounding dialogue from {{char}}, who is {character_name}, featuring an action like a handshake or a playful gesture towards {{user}}. {character_name}'s unique characteristics ({character_summary}, {character_personality}) should shine through. The dialogue should subtly hint at the theme of {topic} without explicit mention. Only {character_name}'s side of the dialogue and actions are to be scripted; do not script {{user}}'s responses and actions. [END INST]")
    print(output)
    return output.strip()

def generate_example_messages(character_name, character_summary, character_personality, topic):
    example_dialogue = ''' 
<s>[INST] Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is {character_name}. {character_name} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what the topic is, and don't describe the topic itself [/INST]
{{user}}: "Good afternoon, Mr. {{char}}. I've heard so much about your success. It's an honor to meet you."
{{char}}: *{{char}} gives a warm smile and extends his hand for a handshake.* "The pleasure is mine, {{user}}. Your reputation precedes you. Let's make this a success together."
{{user}}: *Shakes {{char}}'s hand with a firm grip.* "I look forward to it."
{{char}}: *As they release the handshake, {character_name} leans in, their eyes sharp with interest.* "Impressive. Tell me more about how we align with our goals."
</s>
<s>[INST] Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a playful gesture. Make it sound natural and dynamic. {{char}} is Tatsukaga Yamari. Tatsukaga Yamari characteristics: Tatsukaga Yamari is an anime girl, living in a magical world and solving problems. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself [/INST]
{{user}}: "{{char}}, this place is absolutely enchanting. What's the plan for today?"
{{char}}: *{{char}} grabs {{user}}'s hand and playfully twirls them around before letting go.* "We're off to a new challenge. It's a treacherous journey, but I believe in us."
{{user}}: *Nods with determination.* "With your skills and our friendship, there's nothing we can't accomplish."
{{char}}: *{{char}} moves closer, their eyes shining with trust.* "That's the spirit, {{user}}! Let's embark on this quest together!"
</s>'''

    print("Generating Character Example Messages")
    output = llm(
        example_dialogue + f"\n[INST] Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is {character_name}. {character_name} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself [/INST]\n")
    print(output + "\n")
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
    sd_prompt = args.avatar_prompt if args.avatar_prompt else llm(
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
    images = generate_images(context, prompt=prompt, negative_prompt=negative_prompt or "",
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


def main():
    args = parse_args()
    save_arguments_to_file(args)
    prepare_llm()
    character = create_character(args)
    character_name = character.name.replace(" ", "_")
    if not os.path.exists(character_name):
        os.mkdir(character_name)
    character.export_neutral_json_file(f"{character_name}/{character_name}.json")
    character.export_neutral_yaml_file(f"{character_name}/{character_name}.yml")
    generate_character_avatar(character.name, character.summary, args)
    character.image_path = f"{character_name}/{character_name}.png"
    character.export_neutral_card_file(f"{character_name}/{character_name}.card.png")
    print(character.data_summary)


if __name__ == "__main__":
    main()
