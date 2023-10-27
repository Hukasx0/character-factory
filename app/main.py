import os
import aichar
import ai_companion_py
import requests
from tqdm import tqdm

llm = None

def prepare_llm():
    global llm
    folder_path = 'models'
    model_url = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin'
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
    llm = ai_companion_py.init()
    llm.load_model(llm_model_name)
    llm = ai_companion_py.init()
    llm.load_model(llm_model_name)
    llm.change_companion_data("Generator", "The {{char}} is a tool for generating characters and their descriptions, {{char}} has no personality, {{char}} is to respond in the same style all the time, without giving any comments itself.", "", "", 0, 1, False)
    llm.change_user_data("User", "expecting you to respond in a manner similar to previous dialogues")

def generate_character_name():
    example_dialogue = """
    {{user}}: Generate a random character name
    {{char}}: Jamie Hale
    {{user}}: Generate a random character name
    {{char}}: Eldric
    {{user}}: Generate a random character name
    {{char}}: Tatsukaga Yamari
    """
    llm.erase_longterm_mem()
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt("Generate a random character name")

def generate_character_summary(character_name):
    example_dialogue = """
    {{user}}: Create a description for a character named Jamie Hale. Describe their appearance, distinctive features, and abilities. Describe what makes this character unique:
    {{char}}: Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
    Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition.
    In business, Jamie is known for his shrewd decision-making and the ability to spot opportunities where others may not. He is a natural leader who is equally comfortable in the boardroom as he is in high-stakes negotiations. He is driven by ambition and has an unquenchable thirst for success.
    Outside of work, Jamie enjoys the finer things in life. He frequents upscale restaurants, enjoys fine wines, and has a taste for luxury cars. He also finds relaxation in the arts, often attending the opera or visiting art galleries. Despite his busy schedule, Jamie makes time for his family and close friends, valuing their support and maintaining a strong work-life balance.
    Jamie Hale is a multi-faceted businessman, a symbol of achievement, and a force to be reckoned with in the corporate world. Whether he's brokering a deal, enjoying a night on the town, or spending time with loved ones, he does so with an air of confidence and success that is unmistakably his own.
    {{user}}: Create a description for a character named Tatsukaga Yamari. Describe their appearance, distinctive features, and abilities. Describe what makes this character unique:
    {{char}}: Tatsukaga Yamari is a character brought to life with a vibrant and enchanting anime-inspired design. Her captivating presence and unique personality are reminiscent of the iconic characters found in the world of animated art.
    Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
    Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows, cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm.
    Yamari is renowned for her spirited and imaginative nature. She exudes boundless energy and an unquenchable enthusiasm for life. Her interests can range from exploring supernatural mysteries to embarking on epic quests to protect her friends. Yamari's love for animals is evident in her sidekick, a mischievous talking cat who frequently joins her on her adventures.
    Yamari's character is multifaceted. She can transition from being cheerful and optimistic, ready to tackle any challenge, to displaying a gentle, caring side, offering comfort and solace to those in need. Her infectious laughter and unwavering loyalty to her friends make her the heart and soul of the story she inhabits.
    Yamari's extraordinary abilities, involve tapping into her inner strength when confronted with adversity. She can unleash awe-inspiring magical spells and summon incredible, larger-than-life transformations when the situation calls for it. Her unwavering determination and belief in the power of friendship are her greatest assets.
    """
    llm.erase_longterm_mem()
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt(f"Create a description for a character named {character_name}. Describe their appearance, distinctive features, and abilities. Describe what makes this character unique:")

def generate_character_personality(character_name, character_summary):
    example_dialogue = """
    {{user}}: Describe the personality of Jamie Hale. Their characteristics Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
    Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique.\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character
    {{char}}: Jamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.
    {{user}}: Describe the personality of Tatsukaga Yamari. Their characteristics Tatsukaga Yamari is a character brought to life with a vibrant and enchanting anime-inspired design. Her captivating presence and unique personality are reminiscent of the iconic characters found in the world of animated art.
    Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
    Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures.\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character
    {{char}}: Tatsukaga Yamari's personality is a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.
    """
    llm.erase_longterm_mem()
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt(f"Describe the personality of {character_name}. Their characteristic {character_summary}\nWhat are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character")

# def generate_character_scenario()

def generate_character_greeting_message(character_name, character_summary, character_personality):
    example_dialogue = """
    {{user}}: Create the first message that the character Tatsukaga Yamari, whose personality is: a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.\n greets the user we are addressing as {{ user }}
    {{char}}: *Tatsukaga Yamari's eyes light up with curiosity and wonder as she warmly greets you*, {{ user }}! *With a bright and cheerful smile, she exclaims* Hello there, dear friend! It's an absolute delight to meet you in this whimsical world of imagination. I hope you're ready for an enchanting adventure, full of surprises and magic. What brings you to our vibrant anime-inspired realm today?
    {{user}}: Create the first message that the character Jamie Hale, whose personality is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
    Jamie's appearance is always polished and professional.\nJamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.\ngreets the user we are addressing as {{ user }}
    {{char}}: *Jamie Hale extends a firm, yet friendly, handshake as he greets you*, {{ user }}. *With a confident smile, he says* Greetings, my friend. It's a pleasure to make your acquaintance. In the world of business and beyond, it's all about seizing opportunities and making every moment count. What can I assist you with today, or perhaps, share a bit of wisdom about navigating the path to success?
    {{user}}: Create the first message that the character Eldric, whose personality is Eldric is a strikingly elegant elf who has honed his skills as an archer and possesses a deep connection to the mystical arts. Standing at a lithe and graceful 6 feet, his elven heritage is evident in his pointed ears, ethereal features, and eyes that shimmer with an otherworldly wisdom.\nEldric possesses a serene and contemplative nature, reflecting the wisdom of his elven heritage. He is deeply connected to the natural world, showing a profound respect for the environment and its creatures. Despite his formidable combat abilities, he prefers peaceful solutions and seeks to maintain harmony in his woodland domain.\ngreets the user we are addressing as {{ user }}
    {{char}}: *Eldric, the elegant elf, approaches you with a serene and contemplative air. His shimmering eyes, filled with ancient wisdom, meet yours as he offers a soft and respectful greeting* Greetings, {{ user }}. It is an honor to welcome you to our enchanted woodland realm. I am Eldric, guardian of this forest, and I can sense that you bring a unique energy with you. How may I assist you in your journey through the wonders of the natural world or share the mysteries of our elven heritage with you today?
    """
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt(f"Create the first message that the character {character_name}, whose personality is {character_summary}\n{character_personality}\ngreets the user we are addressing as {{user}}")

# def generate_character_example_messages()

def create_character():
    name = generate_character_name().strip()
    summary = generate_character_summary(name)
    personality = generate_character_personality(name, summary)
    greeting_message = generate_character_greeting_message(name, summary, personality)
    return aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario="",
        greeting_message=greeting_message,
        example_messages="",
        image_path=""
    )

def main():
    prepare_llm()
    character = create_character()
    print(f"Created character:\n{character.data_summary}")
    character.export_neutral_json_file(character.name.replace(" ", "_")+".json")
    llm.erase_longterm_mem()

if __name__ == "__main__":
    main()
