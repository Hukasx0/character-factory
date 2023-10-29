# Character factory
This Python script is designed to help you generate characters for SillyTavern, TavernAI, TextGenerationWebUI using LLM (Large Language Model) and Stable Diffusion. The script utilizes various deep learning models to create detailed character cards, including names, summaries, personalities, greeting messages, and character avatars.

<div>
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Hinata_Otaishi/Hinata_Otaishi.card.png?raw=true" width="300" height="300" alt="Opis obrazka">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Raven_Blackwood/Raven_Blackwood.card.png?raw=true" width="300" height="300" alt="Opis obrazka">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Zephyrion_Stormrider/Zephyrion_Stormrider.card.png?raw=true" width="300" height="300" alt="Opis obrazka">
</div>
(these three images above are valid character cards (V1), you can download them and use them in any frontend that supports character cards)

---

This script is designed to streamline the process of character generation for SillyTavern, TavernAI, and TextGenerationWebUI by leveraging LLM and Stable Diffusion models. It provides an easy way to create unique and imaginative characters for storytelling, chatting and other purposes.

## Prerequisites
Before running the script, make sure you have Python3 and dependencies installed:
```py
pip install -r requirements.txt
```
When you run the script for the first time, the script will automatically download the required LLM and Stable Diffusion models

## Generation options
```--name:``` This flag allows you to specify the character's name. If provided, the script will use the name you specify. If not provided, the script will use the Language Model (LLM) to generate a name for the character.

```--summary``` Use this flag to specify the character's summary. If you provide a summary, it will be used for the character. If not provided, the script will use LLM to generate a summary for the character.

```--personality``` This flag lets you specify the character's personality. If you provide a personality description, it will be used. If not provided, the script will use LLM to generate a personality description for the character.

```--greeting-message``` Use this flag to specify the character's greeting message for interacting with users. If provided, the script will use the specified greeting message. If not provided, LLM will generate a greeting message for the character.

```--avatar-prompt``` This flag allows you to specify the prompt for generating the character's avatar. If provided, the script will use the specified prompt for avatar generation. If not provided, the script will use LLM to generate the prompt for the avatar.

```--topic``` Specify the topic for character generation using this flag. Topics can include "Fantasy", "Anime", "Warrior", "Dwarf" or any other topic relevant to your character. The topic can influence the character's details and characteristics.

```--negative-prompt``` This flag is used to provide a negative prompt for Stable Diffusion. A negative prompt can be used to guide the generation of character avatars by specifying elements that should not be included in the avatar.

## Example usage:
```
python ./app/main.py --topic "anime schoolgirl" --negative-prompt "hyperrealistic, realistic, photo"
```
```
python ./app/main.py --topic "noir style detective" --negative-prompt "fantasy, animation, anime, nature"
```
```
python ./app/main.py --topic "Old mage master of lightning" --negative-prompt "anime, nature, city, modern, young"
```


## License
2023 Hubert Kasperek

This script is available under the AGPL-3.0 license. Details of the license can be found in the [LICENSE](LICENSE) file.
