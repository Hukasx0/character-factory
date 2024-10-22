# WebUI taken from Character Factory WebUI scripts; however, it does not use AI and is intended only for manual editing.

"""

Character Editor
First install dependencies:

pip install gradio
pip install aichar

"""

import os
import gradio as gr
from PIL import Image
import aichar

def save_uploaded_image(image, character_name):
    if image is not None:
        character_name = character_name.replace(" ", "_")
        os.makedirs(f"characters/{character_name}", exist_ok=True)
        image_path = f"characters/{character_name}/{character_name}.png"
        image.save(image_path, format='PNG')
        return image_path
    else:
        return None

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
        raise ValueError("Error when importing character data from a JSON file. Validate the file. Check the file for correctness and try again")

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
        raise ValueError("Error when importing character data from a character card file. Check the file for correctness and try again")

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

def export_character_card(name, summary, personality, scenario, greeting_message, example_messages):
  character_name = name.replace(" ", "_")
  base_path = f"characters/{character_name}/"
  character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=f"{base_path}{character_name}.png"
    )
  character_name = character.name.replace(" ", "_")
  card_path = f"{base_path}{character_name}.card.png"
  character.export_neutral_card_file(card_path)
  return card_path

with gr.Blocks() as webui:
    gr.Markdown("# Character Editor")
    with gr.Tab("Edit character"):
        with gr.Column():
            name = gr.Textbox(placeholder="Character name", label="Name")
            summary = gr.Textbox(placeholder="Character summary", label="Summary", lines=3)
            personality = gr.Textbox(placeholder="Character personality", label="Personality", lines=3)
            scenario = gr.Textbox(placeholder="Character scenario", label="Scenario", lines=3)
            greeting_message = gr.Textbox(placeholder="Character greeting message", label="Greeting Message", lines=2)
            example_messages = gr.Textbox(placeholder="Character example messages", label="Example Messages", lines=4)

            gr.Markdown("## Upload a character avatar (.png file format is recommended)")
            with gr.Row():
                image_input = gr.Image(width=512, height=512, type="pil", sources=["upload", "clipboard"])
                image_input.upload(save_uploaded_image, inputs=[image_input, name], outputs=image_input)

    with gr.Tab("Import character"):
        with gr.Column():
            with gr.Row():
                import_card_input = gr.File(label="Upload character card file", file_types=[".png"])
                import_json_input = gr.File(label="Upload JSON file", file_types=[".json"])
            with gr.Row():
                import_card_button = gr.Button("Import character from character card")
                import_json_button = gr.Button("Import character from json")

            import_card_button.click(
                import_character_card,
                inputs=[import_card_input],
                outputs=[name, summary, personality, scenario, greeting_message, example_messages],
            )
            import_json_button.click(
                import_character_json,
                inputs=[import_json_input],
                outputs=[name, summary, personality, scenario, greeting_message, example_messages],
            )

    with gr.Tab("Export character"):
        with gr.Column():
            with gr.Row():
                export_image = gr.Image(type="pil")
                export_json_textbox = gr.JSON()

            with gr.Row():
                export_card_button = gr.Button("Export as character card")
                export_json_button = gr.Button("Export as JSON")

                export_card_button.click(export_character_card, inputs=[name, summary, personality, scenario, greeting_message, example_messages], outputs=export_image)
                export_json_button.click(
                    export_as_json,
                    inputs=[name, summary, personality, scenario, greeting_message, example_messages],
                    outputs=export_json_textbox,
                )

    gr.HTML("""<div style='text-align: center; font-size: 20px;'>
    <p>
      <a style="text-decoration: none; color: inherit;" href="https://github.com/Hukasx0/character-factory">Character Editor</a>
      by
      <a style="text-decoration: none; color: inherit;" href="https://github.com/Hukasx0">Hubert "Hukasx0" Kasperek</a>
    </p>
  </div>""")

webui.launch(debug=True)
