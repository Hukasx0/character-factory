import os
import urllib.request
import aichar
import ai_companion_py

llm = None

def prepare_llm():
    global llm
    folder_path = 'models'
    model_url = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    llm_model_name = os.path.join(folder_path, os.path.basename(model_url))
    if not os.path.exists(llm_model_name):
        print(f'Downloading LLM model from: {model_url}')
        urllib.request.urlretrieve(model_url, llm_model_name)
    llm = ai_companion_py.init()
    llm.load_model(llm_model_name)
    llm.change_companion_data("Generator", "The {{char}} is a tool for generating characters and their descriptions, {{char}} is to respond in the same style all the time, without giving any comments itself.", "", "", 0, 1, False)
    llm.change_user_data("User", "expecting you to respond in a manner similar to previous dialogues")

def generate_character_name():
    example_dialogue = """
    {{user}}: Generate a random character name
    {{char}}: Jamie Hale
    {{user}}: Generate a random character name
    {{char}}: Isabella Andrews
    {{user}}: Generate a random character name
    {{char}}: Eldric
    """
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt("Generate a random character name")

def generate_character_summary(character_name):
    example_dialogue = """

    """
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt(f"Create a description for a character named {character_name}. Describe their appearance, distinctive features, and abilities. This could be a knight in full armor or an elven character with long silver hair. Describe what makes this character unique:")

def generate_character_personality(character_name):
    example_dialogue = """

    """
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt(f"Describe the personality of {character_name}. Is the character brave and resolute, or perhaps a bit shy? What are their strengths and weaknesses? What values guide this character? Describe them in a way that allows the reader to better understand their character")

# def generate_character_scenario()

def generate_character_greeting_message(character_name):
    example_dialogue = """

    """
    llm.change_companion_example_dialogue(example_dialogue)
    return llm.prompt(f"Create the first message that the character {character_name} greets the user we are addressing as {{user}}")

# def generate_character_example_messages()

def create_character():
    name = generate_character_name().strip()
    summary = generate_character_summary(name)
    personality = generate_character_personality(name)
    greeting_message = generate_character_greeting_message(name)
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

if __name__ == "__main__":
    main()
