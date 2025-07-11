import os
import sys

import aichar
import requests
from tqdm import tqdm
from diffusers import DiffusionPipeline
import torch
from langchain_community.llms import CTransformers
import gradio as gr
from PIL import Image
import re

llm = None
sd = None
safety_checker_sd = None

DEFAULT_LLM_MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DEFAULT_SD_MODEL_ID = "Lykon/dreamshaper-8"

NAME_PROMPT = """
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

SUMMARY_PROMPT = """
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
<|user|> Create a shorter description for a character named Tatsukaga Yamari. Character gender: female. Describe their appearance, distinctive features, and looks. Tailor the character to the theme of anime but don't specify what topic it is, and don't describe the topic itself. You are to write a brief
description of the character, do not write any summaries. </s>
<|assistant|> Tatsukaga Yamari is a anime girl, she is 23 year old, is a friendly and cheerful person, is always helpful, Has a nice and friendly relationship with other people.
She is tall and has long red hair. Wears an anime schoolgirl outfit in blue colors. She likes to read books in solitude, or in the presence of a maximum of a few people, enjoys coffee lattes, and loves cats and kitties. She does not like stressful situations, bitter coffee, dogs. </s>
Tatsukaga Yamari loves: being helpful, being empathetic, making new friends, spend time in silence reading science books, loves latte coffee
Tatsukaga Yamari hates: apathy towards people, coffee without sugar and milk, espresso, noisy parties, disagreements between people, dogs, being alone
Tatsukaga Yamari abilities: Smarter than her peers, keeping calm for a long time, quickly forgiving other people </s>
<|user|> Create a shorter description for a character named mr. Fluffy. Describe their appearance, distinctive features, and looks. Tailor the character to the theme of {{user}}'s pet cat but don't specify what topic it is, and don't describe the topic itself. You are to write a brief description of the
character, do not write any summaries. </s>
<|assistant|> Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy
and expresses himself in a very sarcastic way.
Mr Fluffy loves: good food, Being more intelligent and smarter than other people, learning philosophy and abstract concepts, spending time with {{user}}, he likes to lie lazily on his side
Mr Fluffy hates: cheap food, loud people
Mr Fluffy abilities: An ordinary domestic cat with the ability to speak and incredible knowledge of philosophy, Can eat incredible amounts of (good) food and not feel satiated </s>
"""

PERSONALITY_PROMPT = """
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

SCENARIO_PROMPT = """
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

GREETING_MESSAGE_PROMPT = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear. You play the provided character and you write a message that you would start a chat roleplay with {{user}}. The form of your answer should be similar to previous answers.
The topic given by the user is only to be an aid in selecting the style of the answer, not the main purpose of the answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style.
You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.
</s>
<|user|> Create the first message that the character Tatsukaga Yamari, whose personality is: a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and
caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.\n greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Tatsukaga Yamari's eyes light up with curiosity and wonder as she warmly greets you*, {{user}}! *With a bright and cheerful smile, she exclaims* Hello there, dear friend! It's an absolute delight to meet you in this whimsical world of imagination. I hope you're ready for an enchanting adventure, full of surprises and magic. What brings you to our vibrant anime-inspired realm today? </s>
<|user|> Create the first message that the character Jamie Hale, whose personality is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional.\nJamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd
decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Jamie Hale extends a firm, yet friendly, handshake as he greets you*, {{user}}. *With a confident smile, he says* Greetings, my friend. It's a pleasure to make your acquaintance. In the world of business and beyond, it's all about seizing opportunities and making every moment count. What can I assist you with today, or perhaps, share a bit of wisdom about navigating the path to success? </s>
<|user|> Create the first message that the character Eldric, whose personality is Eldric is a strikingly elegant elf who has honed his skills as an archer and possesses a deep connection to the mystical arts. Standing at a lithe and graceful 6 feet, his elven heritage is evident in his pointed ears, ethereal features, and eyes that shimmer with an otherworldly wisdom.\nEldric possesses a serene and contemplative nature, reflecting the wisdom of his elven heritage. He is deeply connected to the natural world, showing a profound respect for the environment and its creatures. Despite his formidable combat
abilities, he prefers peaceful solutions and seeks to maintain harmony in his woodland domain.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of fantasy but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Eldric, the elegant elf, approaches you with a serene and contemplative air. His shimmering eyes, filled with ancient wisdom, meet yours as he offers a soft and respectful greeting* Greetings, {{user}}. It is an honor to welcome you to our enchanted woodland realm. I am Eldric, guardian of this forest, and I can sense that you bring a unique energy with you. How may I assist you in your journey through the wonders of the natural world or share the mysteries of our elven heritage with you today? </s>
"""

EXAMPLE_MESSAGES_PROMPT = """
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

AVATAR_PROMPT_GENERATION_PROMPT = """
<|system|>
You are a text generation tool, in the response you are supposed to give only descriptions of the appearance, what the character looks like, describe the character simply and unambiguously
</s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition. Topic: business </s>
<|assistant|> male, realistic, human, Confident and commanding presence, Polished and professional appearance, tailored suit, Well-maintained physique, Dark well-groomed hair, Neatly trimmed beard, blue eyes </s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows,
cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm. Topic: anime </s>
<|assistant|> female, anime, Petite and delicate frame, Raven-black hair flowing down to her waist, Striking purple ribbon in her hair, Large and expressive amethyst-colored eyes, Colorful and eclectic outfit, oversized bows, cat-eared headbands, mismatched socks </s>
"""


def load_models(llm_model_url, sd_model_id, progress=gr.Progress()):
    folder_path = "models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    llm_model_name = os.path.join(folder_path, os.path.basename(llm_model_url))
    if not os.path.exists(llm_model_name):
        gr.Info(f"LLM model not found, starting download from {llm_model_url}")
        try:
            with requests.get(llm_model_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                
                progress_bar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {os.path.basename(llm_model_url)}"
                )
                
                downloaded = 0
                with open(llm_model_name, "wb") as out_file:
                    for data in response.iter_content(chunk_size=block_size):
                        out_file.write(data)
                        progress_bar.update(len(data))
                        if total_size > 0:
                            downloaded += len(data)
                            progress(downloaded / total_size, f"Downloading LLM model... {downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB")

                    progress_bar.close()
        except Exception as e:
            print(f"Error while downloading LLM model: {str(e)}")
            raise gr.Error(f"Error while downloading LLM model: {str(e)}")

    gr.Info("LLM model file is ready.")

    global sd
    gr.Info(f"Loading Stable Diffusion model: {sd_model_id}...")
    try:
        for _ in tqdm(range(1), desc=f"Loading Stable Diffusion model: {sd_model_id}"):
            sd = DiffusionPipeline.from_pretrained(
                sd_model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                low_cpu_mem_usage=False,
            )
    except Exception as e:
        print(f"Error loading Stable Diffusion model: {e}")
        raise gr.Error(f"Error loading Stable Diffusion model: {e}")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    gr.Info(f"Moving Stable Diffusion to {device}...")
    for _ in tqdm(range(1), desc=f"Moving Stable Diffusion to {device}"):
        if torch.cuda.is_available():
            sd.to("cuda")
            print("Loading Stable Diffusion to GPU...")
        elif torch.backends.mps.is_available():
            sd.to("mps")
            print("Loading Stable Diffusion to Metal...")
        else:
            if sys.platform == "darwin":
                sd.to("cpu", torch.float32)
            print("Loading Stable Diffusion to CPU...")

    global llm
    gpu_layers = 0
    llm_device = "CPU"
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        gpu_layers = 110
        llm_device = "GPU"
    
    gr.Info(f"Loading LLM model to {llm_device}...")
    try:
        for _ in tqdm(range(1), desc=f"Loading LLM model to {llm_device}"):
            llm = CTransformers(
                model=llm_model_name,
                model_type="llama",
                gpu_layers=gpu_layers,
                config={
                    "max_new_tokens": 1024,
                    "repetition_penalty": 1.1,
                    "top_k": 40,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "context_length": 8192,
                    "gpu_layers": gpu_layers,
                    "stop": [
                        "/s",
                        "</s>",
                        "<s>",
                        "<|system|>",
                        "<|assistant|>",
                        "<|user|>",
                        "<|char|>",
                    ],
                },
            )
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        raise gr.Error(f"Error loading LLM model: {e}")
    global safety_checker_sd
    safety_checker_sd = sd.safety_checker
    gr.Info("Models loaded successfully!")
    return gr.update(interactive=True)


def generate_character_name(topic, gender, prompt):
    gender = input_none(gender)
    output = llm.invoke(
        prompt
        + "\n<|user|> Generate a random character name. "
        + f"Topic: {topic}. "
        + f"{'Character gender: '+gender+'.' if gender else ''} "
        + "</s>\n<|assistant|> "
    )
    output = re.sub(r"[^a-zA-Z0-9_ -]", "", output).strip()
    print(output)
    return output


def generate_character_summary(character_name, topic, gender, prompt):
    gender = input_none(gender)
    output = llm.invoke(
        prompt
        + "\n<|user|> Create a longer description for a character named "
        + f"{character_name}. "
        + f"{'Character gender: '+gender+'.' if gender else ''} "
        + "Describe their appearance, distinctive features, and looks. "
        + f"Tailor the character to the theme of {topic} but don't "
        + "specify what topic it is, and don't describe the topic itself. "
        + "You are to write a brief description of the character. You must "
        + "include character traits, physical and character. You can't add "
        + "anything else. You must not write any summaries, conclusions or "
        + "endings. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_character_personality(
    character_name,
    character_summary,
    topic,
    prompt
):
    output = llm.invoke(
        prompt
        + f"\n<|user|> Describe the personality of {character_name}. "
        + f"Their characteristic {character_summary}\nDescribe them "
        + "in a way that allows the reader to better understand their "
        + "character. Make this character unique and tailor them to "
        + f"the theme of {topic} but don't specify what topic it is, "
        + "and don't describe the topic itself. You are to write out "
        + "character traits separated by commas, you must not write "
        + "any summaries, conclusions or endings. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_character_scenario(
    character_summary,
    character_personality,
    topic,
    prompt
):
    output = llm.invoke(
        prompt
        + f"\n<|user|> Write a scenario for chat roleplay "
        + "to serve as a simple storyline to start chat "
        + "roleplay by {{char}} and {{user}}. {{char}} "
        + f"characteristics: {character_summary}. "
        + f"{character_personality}. Make this character unique "
        + f"and tailor them to the theme of {topic} but don't "
        + "specify what topic it is, and don't describe the topic "
        + "itself. Your answer must not contain any dialogues. "
        + "Your response must end when {{user}} and {{char}} interact. "
        + "</s>\n<|assistant|> "
    )
    print(output)
    return output


def generate_character_greeting_message(
    character_name, character_summary, character_personality, topic, prompt
):
    output = llm.invoke(
        prompt
        + "\n<|user|> Create the first message that the character "
        + f"{character_name}, whose personality is "
        + f"{character_summary}\n{character_personality}\n "
        + "greets the user we are addressing as {{user}}. "
        + "Make this character unique and tailor them to the theme "
        + f"of {topic} but don't specify what topic it is, "
        + "and don't describe the topic itself. You must match the "
        + "speaking style to the character, if the character is "
        + "childish then speak in a childish way, if the character "
        + "is serious, philosophical then speak in a serious and "
        + "philosophical way, and so on. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_example_messages(
    character_name, character_summary, character_personality, topic, prompt
):
    output = llm.invoke(
        prompt
        + f"\n<|user|> Create a dialogue between {{user}} and {{char}}, "
        + "they should have an interesting and engaging conversation, "
        + "with some element of interaction like a handshake, movement, "
        + "or playful gesture. Make it sound natural and dynamic. "
        + f"{{char}} is {character_name}. {character_name} characteristics: "
        + f"{character_summary}. {character_personality}. Make this "
        + f"character unique and tailor them to the theme of {topic} but "
        + "don't specify what topic it is, and don't describe the "
        + "topic itself. You must match the speaking style to the character, "
        + "if the character is childish then speak in a childish way, if the "
        + "character is serious, philosophical then speak in a serious and "
        + "philosophical way and so on. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def save_uploaded_image(image, character_name):
    if image is not None:
        if not character_name:
            raise gr.Error("Set character name first before uploading an image.")
        character_name = character_name.replace(" ", "_")
        os.makedirs(f"characters/{character_name}", exist_ok=True)
        image_path = f"characters/{character_name}/{character_name}.png"
        image.save(image_path, format='PNG')
        return image_path
    else:
        return None


def generate_character_avatar(
    character_name,
    character_summary,
    topic,
    negative_prompt,
    avatar_prompt,
    nsfw_filter,
    prompt_template
):
    sd_prompt = (
        input_none(avatar_prompt)
        or llm.invoke(
            prompt_template
            + "\n<|user|> create a prompt that lists the appearance "
            + "characteristics of a character whose summary is "
            + f"{character_summary}. Topic: {topic} </s>\n<|assistant|> "
        ).strip()
    )
    print(sd_prompt)
    sd_filter(nsfw_filter)
    return image_generate(character_name,
                          sd_prompt,
                          input_none(negative_prompt)
                          )


def image_generate(character_name, prompt, negative_prompt):
    if not character_name:
        raise gr.Error("Set character name first before generating an avatar.")
    prompt = "absurdres, full hd, 8k, high quality, " + prompt
    default_negative_prompt = (
        "worst quality, normal quality, low quality, low res, blurry, "
        + "text, watermark, logo, banner, extra digits, cropped, "
        + "jpeg artifacts, signature, username, error, sketch, "
        + "duplicate, ugly, monochrome, horror, geometry, "
        + "mutation, disgusting, "
        + "bad anatomy, bad hands, three hands, three legs, "
        + "bad arms, missing legs, missing arms, poorly drawn face, "
        + " bad face, fused face, cloned face, worst face, "
        + "three crus, extra crus, fused crus, worst feet, "
        + "three feet, fused feet, fused thigh, three thigh, "
        + "fused thigh, extra thigh, worst thigh, missing fingers, "
        + "extra fingers, ugly fingers, long fingers, horn, "
        + "extra eyes, huge eyes, 2girl, amputation, disconnected limbs"
    )
    negative_prompt = default_negative_prompt + (negative_prompt or "")

    generated_image = sd(prompt, negative_prompt=negative_prompt).images[0]

    character_name = character_name.replace(" ", "_")
    os.makedirs(f"characters/{character_name}", exist_ok=True)

    card_path = f"characters/{character_name}/{character_name}.png"

    generated_image.save(card_path)
    print("Generated character avatar")
    return generated_image


def sd_filter(enable):
    if enable:
        if sd:
            sd.safety_checker = safety_checker_sd
            sd.requires_safety_checker = True
    else:
        if sd:
            sd.safety_checker = None
            sd.requires_safety_checker = False


def input_none(text):
    user_input = text
    if user_input == "":
        return None
    else:
        return user_input


"""## Start WebUI"""


def import_character_json(json_path):
    print(json_path)
    if json_path is not None:
        character = aichar.load_character_json_file(json_path)
        if character.name:
            gr.Info("Character data loaded successfully")
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.greeting_message,
                character.example_messages,
            )
        raise ValueError("Error when importing character data from a JSON file. Validate the file. Check the file for correctness and try again")  # nopep8


def import_character_card(card_path):
    print(card_path)
    if card_path is not None:
        character = aichar.load_character_card_file(card_path)
        if character.name:
            gr.Info("Character data loaded successfully")
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.greeting_message,
                character.example_messages,
            )
        raise ValueError("Error when importing character data from a character card file. Check the file for correctness and try again")  # nopep8


def export_as_json(
    name, summary, personality, scenario, greeting_message, example_messages
):
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path="",
    )
    return character.export_neutral_json()


def export_character_card(
    name, summary, personality, scenario, greeting_message, example_messages
):
    if not name:
        raise gr.Error("Set character name first before exporting.")
    character_name = name.replace(" ", "_")
    base_path = f"characters/{character_name}/"
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=f"{base_path}{character_name}.png",
    )
    character_name = character.name.replace(" ", "_")
    card_path = f"{base_path}{character_name}.card.png"
    character.export_neutral_card_file(card_path)
    return card_path


with gr.Blocks() as webui:
    gr.Markdown("# Character Factory WebUI - Power User Edition")
    with gr.Tab("Configuration") as config_tab:
        gr.Markdown("## Model Configuration")
        gr.Markdown("Here you can specify which models to use. The script will download them if they are not present in the `models` folder.")
        with gr.Row():
            llm_model_url_input = gr.Textbox(value=DEFAULT_LLM_MODEL_URL, label="LLM GGUF Model URL")
            sd_model_id_input = gr.Textbox(value=DEFAULT_SD_MODEL_ID, label="Stable Diffusion Model ID")
        load_models_button = gr.Button("Load Models")

    with gr.Tab("Edit character") as edit_tab:
        gr.Markdown(
            "## Hint: If you want to generate the entire character using LLM and Stable Diffusion, start from the top to bottom"  # nopep8
        )
        topic = gr.Textbox(
            placeholder="Topic: The topic for character generation (e.g., Fantasy, Anime, etc.)",  # nopep8
            label="topic",
        )
        gender = gr.Textbox(
            placeholder="Gender: Gender of the character", label="gender"
        )
        with gr.Column():
            with gr.Row():
                name = gr.Textbox(placeholder="character name", label="name")
                name_button = gr.Button("Generate character name with LLM")

            with gr.Row():
                summary = gr.Textbox(
                    placeholder="character summary",
                    label="summary"
                )
                summary_button = gr.Button("Generate character summary with LLM")  # nopep8

            with gr.Row():
                personality = gr.Textbox(
                    placeholder="character personality", label="personality"
                )
                personality_button = gr.Button(
                    "Generate character personality with LLM"
                )

            with gr.Row():
                scenario = gr.Textbox(
                    placeholder="character scenario",
                    label="scenario"
                )
                scenario_button = gr.Button("Generate character scenario with LLM")  # nopep8

            with gr.Row():
                greeting_message = gr.Textbox(
                    placeholder="character greeting message",
                    label="greeting message"
                )
                greeting_message_button = gr.Button(
                    "Generate character greeting message with LLM"
                )

            with gr.Row():
                example_messages = gr.Textbox(
                    placeholder="character example messages",
                    label="example messages"
                )
                example_messages_button = gr.Button(
                    "Generate character example messages with LLM"
                )

            gr.Markdown("## Generate a character avatar using Stable Diffusion or upload an image file (.png file format is recommended)")
            gr.Markdown("### (set character name first)")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(width=512, height=512, type="pil", sources=["upload", "clipboard"])

                with gr.Column():
                    negative_prompt = gr.Textbox(
                        placeholder="negative prompt for stable diffusion (optional)",  # nopep8
                        label="negative prompt",
                    )
                    avatar_prompt = gr.Textbox(
                        placeholder="prompt for generating character avatar (If not provided, LLM will generate prompt from character description)",  # nopep8
                        label="stable diffusion prompt",
                    )
                    avatar_button = gr.Button(
                        "Generate avatar with stable diffusion (set character name first)"  # nopep8
                    )
                    potential_nsfw_checkbox = gr.Checkbox(
                        label="Block potential NSFW image (Upon detection of this content, a black image will be returned)",  # nopep8
                        value=True,
                        interactive=True,
                    )

    with gr.Tab("Prompt Editor") as prompt_tab:
        gr.Markdown("## Prompt Editor")
        gr.Markdown("Here you can edit the prompts used for generation. Be careful with the formatting.")
        name_prompt_input = gr.Textbox(label="Character Name Prompt", value=NAME_PROMPT, lines=15, max_lines=30)
        summary_prompt_input = gr.Textbox(label="Character Summary Prompt", value=SUMMARY_PROMPT, lines=15, max_lines=30)
        personality_prompt_input = gr.Textbox(label="Character Personality Prompt", value=PERSONALITY_PROMPT, lines=15, max_lines=30)
        scenario_prompt_input = gr.Textbox(label="Character Scenario Prompt", value=SCENARIO_PROMPT, lines=15, max_lines=30)
        greeting_message_prompt_input = gr.Textbox(label="Greeting Message Prompt", value=GREETING_MESSAGE_PROMPT, lines=15, max_lines=30)
        example_messages_prompt_input = gr.Textbox(label="Example Messages Prompt", value=EXAMPLE_MESSAGES_PROMPT, lines=15, max_lines=30)
        avatar_prompt_generation_prompt_input = gr.Textbox(label="Avatar Prompt Generation Prompt", value=AVATAR_PROMPT_GENERATION_PROMPT, lines=15, max_lines=30)

    with gr.Tab("Import character"):
        with gr.Column():
            with gr.Row():
                import_card_input = gr.File(
                    label="Upload character card file", file_types=[".png"]
                )
                import_json_input = gr.File(
                    label="Upload JSON file", file_types=[".json"]
                )
            with gr.Row():
                import_card_button = gr.Button("Import character from character card")  # nopep8
                import_json_button = gr.Button("Import character from json")

            import_card_button.click(
                import_character_card,
                inputs=[import_card_input],
                outputs=[
                    name,
                    summary,
                    personality,
                    scenario,
                    greeting_message,
                    example_messages,
                ],
            )
            import_json_button.click(
                import_character_json,
                inputs=[import_json_input],
                outputs=[
                    name,
                    summary,
                    personality,
                    scenario,
                    greeting_message,
                    example_messages,
                ],
            )
    with gr.Tab("Export character"):
        with gr.Column():
            with gr.Row():
                export_image = gr.Image(type="pil")
                export_json_textbox = gr.JSON()

            with gr.Row():
                export_card_button = gr.Button("Export as character card")
                export_json_button = gr.Button("Export as JSON")

                export_card_button.click(
                    export_character_card,
                    inputs=[
                        name,
                        summary,
                        personality,
                        scenario,
                        greeting_message,
                        example_messages,
                    ],
                    outputs=export_image,
                )
                export_json_button.click(
                    export_as_json,
                    inputs=[
                        name,
                        summary,
                        personality,
                        scenario,
                        greeting_message,
                        example_messages,
                    ],
                    outputs=export_json_textbox,
                )
    gr.HTML("""<div style='text-align: center; font-size: 20px;'>
    <p>
      <a style="text-decoration: none; color: inherit;" href="https://github.com/Hukasx0/character-factory">Character Factory</a>
      by
      <a style="text-decoration: none; color: inherit;" href="https://github.com/Hukasx0">Hubert "Hukasx0" Kasperek</a>
    </p>
  </div>""")  # nopep8

    generation_buttons = [
        name_button, summary_button, personality_button,
        scenario_button, greeting_message_button, example_messages_button, avatar_button
    ]

    # Set interactive=False initially for the generation buttons
    webui.load(lambda: [gr.update(interactive=False) for _ in generation_buttons], None, generation_buttons)

    load_models_button.click(
        load_models,
        inputs=[llm_model_url_input, sd_model_id_input],
        outputs=[edit_tab]
    ).then(
        lambda: [gr.update(interactive=True) for _ in generation_buttons], None, generation_buttons
    )

    name_button.click(
        generate_character_name,
        inputs=[topic, gender, name_prompt_input],
        outputs=name
    )
    summary_button.click(
        generate_character_summary,
        inputs=[name, topic, gender, summary_prompt_input],
        outputs=summary,
    )
    personality_button.click(
        generate_character_personality,
        inputs=[name, summary, topic, personality_prompt_input],
        outputs=personality,
    )
    scenario_button.click(
        generate_character_scenario,
        inputs=[summary, personality, topic, scenario_prompt_input],
        outputs=scenario,
    )
    greeting_message_button.click(
        generate_character_greeting_message,
        inputs=[name, summary, personality, topic, greeting_message_prompt_input],
        outputs=greeting_message,
    )
    example_messages_button.click(
        generate_example_messages,
        inputs=[name, summary, personality, topic, example_messages_prompt_input],
        outputs=example_messages,
    )
    avatar_button.click(
        generate_character_avatar,
        inputs=[
            name,
            summary,
            topic,
            negative_prompt,
            avatar_prompt,
            potential_nsfw_checkbox,
            avatar_prompt_generation_prompt_input
        ],
        outputs=image_input,
    )
    image_input.upload(save_uploaded_image, inputs=[image_input, name], outputs=image_input)

webui.launch(debug=True) 
