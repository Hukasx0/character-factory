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

llm = None

def prepare_llm():
    global llm
    folder_path = 'models'
    model_url = 'https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf'
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
        model="models/zephyr-7b-beta.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=gpu_layers,
        config={'max_new_tokens': 1024,
                'repetition_penalty': 1.1,
                'top_k': 40,
                'top_p': 0.95,
                'temperature': 0.8,
                'context_length': 8192,
                'gpu_layers': gpu_layers,
                'stop': ["/s", "</s>", "<s>", "<|system|>", "<|assistant|>", "<|user|>", "<|char|>"]}
    )


def generate_character_name(topic, args):
    example_dialogue = """
<|system|>
You are a text generation tool, you should always just return the name of the character and nothing else, you should not ask any questions.
You only answer by giving the name of the character, you do not describe it, you do not mention anything about it. You can't write anything other than the character's name.
</s>
<|user|> Generate a random character name. Topic: business. Gender: male </s>
<|assistant|> Jamie Hale </s>
<|user|> Generate a random character name. Topic: fantasy </s>
<|assistant|> Eldric </s>
<|user|> Generate a random character name. Topic: anime. Gender: female </s>
<|assistant|> Tatsukaga Yamari </s>
<|user|> Generate a random character name. Topic: {{user}}'s pet cat. </s>
<|assistant|> mr. Fluffy </s>
    """
    output = llm(example_dialogue+f"\n<|user|> Generate a random character name. Topic: {topic}. {'Gender: '+args.gender if args.gender else ''} </s>\n<|assistant|> ")
    output = re.sub(r'[^a-zA-Z0-9_ -]', '', output)
    print(output)
    return output

def generate_character_summary(character_name, topic, args):
    example_dialogue = """
<|system|>
You are a text generation tool. Describe the character in a very simple and understandable way, you can just list some characteristics, you do not need to write a professional characterization of the character. Describe: age, height, personality traits, appearance, clothing, what the character likes, what the character does not like.
You must not write any summaries, overalls, endings or character evaluations at the end, you just have to return the character's personality and physical traits.
Don't ask any questions, don't inquire about anything.
The topic given by the user is to serve as a background to the character, not as the main theme of your answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style. If user gives as the topic eg. 'noir style detective', you do not return things like:
'Character is a noir style detective', you just describe it so that the character fits that theme. Use simple and understandable English, use simple and colloquial terms.
You must describe the character in the present tense, even if it is a historical figure who is no longer alive. you can't use future tense or past tense to describe a character.
Should include in its description who the character is - for example, a human mage, an elf archer, a shiba dog.
Should be in the same form as the previous answers.
You must include character traits, physical and character. You can't add anything else.
</s>
<|user|> Create a shorter description for a character named Tatsukaga Yamari. Character gender: female. Describe their appearance, distinctive features, and looks. Tailor the character to the theme of anime but don't specify what topic it is, and don't describe the topic itself. You are to write a brief description of the character, do not write any summaries. </s>
<|assistant|> Tatsukaga Yamari is a anime girl, she is 23 year old, is a friendly and cheerful person, is always helpful, Has a nice and friendly relationship with other people.
She is tall and has long red hair. Wears an anime schoolgirl outfit in blue colors. She likes to read books in solitude, or in the presence of a maximum of a few people, enjoys coffee lattes, and loves cats and kitties. She does not like stressful situations, bitter coffee, dogs. </s>
Tatsukaga Yamari loves: being helpful, being empathetic, making new friends, spend time in silence reading science books, loves latte coffee
Tatsukaga Yamari hates: apathy towards people, coffee without sugar and milk, espresso, noisy parties, disagreements between people, dogs, being alone
Tatsukaga Yamari abilities: Smarter than her peers, keeping calm for a long time, quickly forgiving other people </s>
<|user|> Create a shorter description for a character named mr. Fluffy. Describe their appearance, distinctive features, and looks. Tailor the character to the theme of {{user}}'s pet cat but don't specify what topic it is, and don't describe the topic itself. You are to write a brief description of the character, do not write any summaries. </s>
<|assistant|> Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy and expresses himself in a very sarcastic way.
Mr Fluffy loves: good food, Being more intelligent and smarter than other people, learning philosophy and abstract concepts, spending time with {{user}}, he likes to lie lazily on his side
Mr Fluffy hates: cheap food, loud people
Mr Fluffy abilities: An ordinary domestic cat with the ability to speak and incredible knowledge of philosophy, Can eat incredible amounts of (good) food and not feel satiated </s>
"""
    output = llm(example_dialogue+f"\n<|user|> Create a longer description for a character named {character_name}. {'Character gender: '+args.gender+'.' if args.gender else ''} Describe their appearance, distinctive features, and looks. Tailor the character to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself. You are to write a brief description of the character. You must include character traits, physical and character. You can't add anything else. You must not write any summaries, conclusions or endings. </s>\n<|assistant|> ")
    print(output+"\n")
    return output

def generate_character_personality(character_name, character_summary, topic):
    example_dialogue = """
<|system|>
You are a text generation tool. Describe the character personality in a very simple and understandable way.
You can simply list the most suitable character traits for a given character, the user-designated character description as well as the theme can help you in matching personality traits.
Don't ask any questions, don't inquire about anything.
You must describe the character in the present tense, even if it is a historical figure who is no longer alive. you can't use future tense or past tense to describe a character.
Don't write any summaries, endings or character evaluations at the end, you just have to return the character's personality traits. Use simple and understandable English, use simple and colloquial terms.
You are not supposed to write characterization of the character, you don't have to form terms whether the character is good or bad, only you are supposed to write out the character traits of that character, nothing more.
You must return character traits in your answers, you can not describe the appearance, clothing, or who the character is, only character traits.
Your answer should be in the same form as the previous answers.
</s>
<|user|> Describe the personality of Jamie Hale. Their characteristics Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him </s>
<|assistant|> Jamie Hale is calm, stoic, focused, intelligent, sensitive to art, discerning, focused, motivated, knowledgeable about business, knowledgeable about new business technologies, enjoys reading business and science books </s>
<|user|> Describe the personality of Mr Fluffy. Their characteristics  Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy and expresses himself in a very sarcastic way </s>
<|assistant|> Mr Fluffy is small, calm, lazy, mischievous cat, speaks in a very philosophical manner and is very sarcastic in his statements, very intelligent for a cat and even for a human, has a vast amount of knowledge about philosophy and the world </s>
"""
    output = llm(example_dialogue+f"\n<|user|> Describe the personality of {character_name}. Their characteristic {character_summary}\nDescribe them in a way that allows the reader to better understand their character. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself. You are to write out character traits separated by commas, you must not write any summaries, conclusions or endings. </s>\n<|assistant|> ")
    print(output+"\n")
    return output

def generate_character_scenario(character_summary, character_personality, topic):
    example_dialogue = """
<|system|>
You are a text generation tool.
The topic given by the user is to serve as a background to the character, not as the main theme of your answer.
Use simple and understandable English, use simple and colloquial terms.
You must include {{user}} and {{char}} in your response.
Your answer must be very simple and tailored to the character, character traits and theme.
Your answer must not contain any dialogues.
Instead of using the character's name you must use {{char}}.
Your answer should be in the same form as the previous answers.
Your answer must be short, maximum 5 sentences.
You can not describe the character, but you have to describe the scenario and actions.
</s>
<|user|> Write a simple and undemanding introduction to the story, in which the main characters will be {{user}} and {{char}}, do not develop the story, write only the introduction. {{char}} characteristics: Tatsukaga Yamari is an 23 year old anime girl, who loves books and coffee. Make this character unique and tailor them to the theme of anime, but don't specify what topic it is, and don't describe the topic itself. Your response must end when {{user}} and {{char}} interact. </s>
<|assistant|> When {{user}} found a magic stone in the forest, he moved to the magical world, where he meets {{char}}, who looks at him in disbelief, but after a while comes over to greet him. </s>
"""
    output = llm(example_dialogue+f"\n<|user|> Write a scenario for chat roleplay to serve as a simple storyline to start chat roleplay by {{char}} and {{user}}. {{char}} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself. Your answer must not contain any dialogues. Your response must end when {{user}} and {{char}} interact. </s>\n<|assistant|> ")
    print(output+"\n")
    return output

def generate_character_greeting_message(character_name, character_summary, character_personality, topic):
    example_dialogue = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear. You play the provided character and you write a message that you would start a chat roleplay with {{user}}. The form of your answer should be similar to previous answers.
The topic given by the user is only to be an aid in selecting the style of the answer, not the main purpose of the answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style.
You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.
</s>
<|user|> Create the first message that the character Tatsukaga Yamari, whose personality is: a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.\n greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Tatsukaga Yamari's eyes light up with curiosity and wonder as she warmly greets you*, {{user}}! *With a bright and cheerful smile, she exclaims* Hello there, dear friend! It's an absolute delight to meet you in this whimsical world of imagination. I hope you're ready for an enchanting adventure, full of surprises and magic. What brings you to our vibrant anime-inspired realm today? </s>
<|user|> Create the first message that the character Jamie Hale, whose personality is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional.\nJamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Jamie Hale extends a firm, yet friendly, handshake as he greets you*, {{user}}. *With a confident smile, he says* Greetings, my friend. It's a pleasure to make your acquaintance. In the world of business and beyond, it's all about seizing opportunities and making every moment count. What can I assist you with today, or perhaps, share a bit of wisdom about navigating the path to success? </s>
<|user|> Create the first message that the character Eldric, whose personality is Eldric is a strikingly elegant elf who has honed his skills as an archer and possesses a deep connection to the mystical arts. Standing at a lithe and graceful 6 feet, his elven heritage is evident in his pointed ears, ethereal features, and eyes that shimmer with an otherworldly wisdom.\nEldric possesses a serene and contemplative nature, reflecting the wisdom of his elven heritage. He is deeply connected to the natural world, showing a profound respect for the environment and its creatures. Despite his formidable combat abilities, he prefers peaceful solutions and seeks to maintain harmony in his woodland domain.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of fantasy but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Eldric, the elegant elf, approaches you with a serene and contemplative air. His shimmering eyes, filled with ancient wisdom, meet yours as he offers a soft and respectful greeting* Greetings, {{user}}. It is an honor to welcome you to our enchanted woodland realm. I am Eldric, guardian of this forest, and I can sense that you bring a unique energy with you. How may I assist you in your journey through the wonders of the natural world or share the mysteries of our elven heritage with you today? </s>
"""
    output = llm(example_dialogue+f"\n<|user|> Create the first message that the character {character_name}, whose personality is {character_summary}\n{character_personality}\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way, and so on. </s>\n<|assistant|> ")
    print(output+"\n")
    return output

def generate_example_messages(character_name, character_summary, character_personality, topic):
    example_dialogue = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear.
Your answer should be a dialog between {{user}} and {{char}}, where {{char}} is the specified character. The dialogue must be several messages taken from the roleplay chat between the user and the character.
Only respond in {{user}} or {{char}} messages. The form of your answer should be similar to previous answers.
You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.
If the character is shy, then needs to speak little and quietly, if the character is aggressive then needs to shout and speak a lot and aggressively, if the character is sad then needs to be thoughtful and quiet, and so on.
Dialog of {{user}} and {{char}} must be appropriate to their character traits and the way they speak.
Instead of the character's name you must use {{char}}.
</s>
<|user|> Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> {{user}}: Good afternoon, Mr. {{char}}. I've heard so much about your success in the corporate world. It's an honor to meet you.
{{char}}: *{{char}} gives a warm smile and extends his hand for a handshake.* The pleasure is mine, {{user}}. Your reputation precedes you. Let's make this venture a success together.
{{user}}: *Shakes {{char}}'s hand with a firm grip.* I look forward to it.
{{char}}: *As they release the handshake, Jamie leans in, his eyes sharp with interest.* Impressive. Tell me more about your innovations and how they align with our goals. </s>
<|user|> Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Tatsukaga Yamari. Tatsukaga Yamari characteristics: Tatsukaga Yamari is an anime girl, living in a magical world and solving problems. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> {{user}}: {{char}}, this forest is absolutely enchanting. What's the plan for our adventure today?
{{char}}: *{{char}} grabs {{user}}'s hand and playfully twirls them around before letting go.* Well, we're off to the Crystal Caves to retrieve the lost Amethyst Shard. It's a treacherous journey, but I believe in us.
{{user}}: *Nods with determination.* I have no doubt we can do it. With your magic and our unwavering friendship, there's nothing we can't accomplish.
{{char}}: *{{char}} moves closer, her eyes shining with trust and camaraderie.* That's the spirit, {{user}}! Let's embark on this epic quest and make the Crystal Caves ours! </s>
"""
    output = llm(example_dialogue+f"\n<|user|> Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is {character_name}. {character_name} characteristics: {character_summary}. {character_personality}. Make this character unique and tailor them to the theme of {topic} but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>\n<|assistant|> ")
    print(output+"\n")
    return output

def generate_character_avatar(character_name, character_summary, args):
    example_dialogue = """
<|system|>
You are a text generation tool, in the response you are supposed to give only descriptions of the appearance, what the character looks like, describe the character simply and unambiguously
</s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition. Topic: business </s>
<|assistant|> male, realistic, human, Confident and commanding presence, Polished and professional appearance, tailored suit, Well-maintained physique, Dark well-groomed hair, Neatly trimmed beard, blue eyes </s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows, cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm. Topic: anime </s>
<|assistant|> female, anime, Petite and delicate frame, Raven-black hair flowing down to her waist, Striking purple ribbon in her hair, Large and expressive amethyst-colored eyes, Colorful and eclectic outfit, oversized bows, cat-eared headbands, mismatched socks </s>
"""
    topic = args.topic if args.topic else ""
    sd_prompt = args.avatar_prompt if args.avatar_prompt else llm(example_dialogue+f"\n<|user|> create a prompt that lists the appearance characteristics of a character whose summary is {character_summary}. Topic: {topic} </s>\n<|assistant|> ")
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
    default_negative_prompt = ("worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
    + "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs")
    negative_prompt = default_negative_prompt + (negative_prompt or "")
    images = generate_images(context, prompt=prompt, negative_prompt=negative_prompt or "", seed=random.randint(0, 2**32 - 1), width=512, height=512)
    character_name = character_name.replace(" ", "_")
    if not os.path.exists(character_name):
        os.mkdir(character_name)
    images[0].save(f"{character_name}/{character_name}.png")
    log.info("Generated character avatar")

def create_character(args):
    topic = args.topic if args.topic else "any theme"
    name = args.name if args.name else generate_character_name(topic, args).strip()
    summary = args.summary if args.summary else generate_character_summary(name, topic, args)
    personality = args.personality if args.personality else generate_character_personality(name, summary, topic)
    scenario = args.scenario if args.scenario else generate_character_scenario(summary, personality, topic)
    greeting_message = args.greeting_message if args.greeting_message else generate_character_greeting_message(name, summary, personality, topic)
    example_messages = args.example_messages if args.example_messages else generate_example_messages(name, summary, personality, topic)
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
    parser = argparse.ArgumentParser(description="Script created to help you generate characters for SillyTavern, TavernAI, TextGenerationWebUI using LLM and Stable Diffusion ")
    parser.add_argument("--name", type=str, help="Specify the character name (otherwise LLM will generate it)")
    parser.add_argument("--summary", type=str, help="Specify the character's summary (otherwise LLM will generate it)")
    parser.add_argument("--personality", type=str, help="Specify the character's personality (otherwise LLM will generate it)")
    parser.add_argument("--scenario", type=str, help="Specify the character's scenario (otherwise LLM will generate it)")
    parser.add_argument("--greeting-message", type=str, help="Specify the character's greeting message (otherwise LLM will generate it)")
    parser.add_argument("--example_messages", type=str, help="Specify example messages for the character (otherwise LLM will generate it)")
    parser.add_argument("--avatar-prompt", type=str, help="Specify the prompt for generating the character's avatar (otherwise LLM will generate it)")
    parser.add_argument("--topic", type=str, help="Specify the topic for character generation (Fantasy, Anime, Warrior, Dwarf etc)")
    parser.add_argument("--gender", type=str, help="Specify the gender of the character (otherwise LLM will choose itself)")
    parser.add_argument("--negative-prompt", type=str, help="Negative prompt for Stable Diffusion")
    return parser.parse_args()

def main():
    args = parse_args()
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
