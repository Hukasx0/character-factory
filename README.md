# Character factory With oobabooga/text-generation-webui and (not much tested) llama.ccp SUPPORT.

<a target="_blank" href="https://colab.research.google.com/drive/1JqkrtFXKalcmuMvST2VltoS1UVwoQINH">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This Python script is designed to help you generate characters for [SillyTavern](https://github.com/SillyTavern/SillyTavern), [TavernAI](https://github.com/TavernAI/TavernAI), [TextGenerationWebUI](https://github.com/oobabooga/text-generation-webui) and many more, using LLM (Large Language Model) and Stable Diffusion. The script utilizes various deep learning models to create detailed character cards, including names, summaries, personalities, greeting messages, and character avatars.

<div>
  My Characters:
  <img src="https://github.com/thijsi123/character-factory/blob/75d1de7cf0aac46c141429fa105fb7bc8985ec18/examples/Kai_Akashi/Kai_Akashi.card.png?raw=true" width="300" height="300" alt="Zara_Moonstone card">
  <img src="https://github.com/thijsi123/character-factory/blob/0647d43549c3e9bf6417ebdec9a19164d9263763/examples/Kaito_Kuroba/Kaito_Kuroba.card.png?raw=true" width="300" height="300" alt="Kaito Kuroba card">
  Original characters:
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Neko_Yui_Ichinose/Neko_Yui_Ichinose.card.png?raw=true" width="300" height="300" alt="Neko Yui Ichinose character card">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Cassandra_Vargas/Cassandra_Vargas.card.png?raw=true" width="300" height="300" alt="Cassandra Vargas character card">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Arthondt_Lightbringer/Arthondt_Lightbringer.card.png?raw=true" width="300" height="300" alt="Arthondt Lightbringer character card">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Albert_Einstein/Albert_Einstein.card.png?raw=true" width="300" height="300" alt="Albert Einstein character card">
  
</div>
(these four images above are valid character cards (V1), you can download them and use them in any frontend that supports character cards)

---

This script is designed to streamline the process of character generation for SillyTavern, TavernAI, and TextGenerationWebUI by leveraging LLM and Stable Diffusion models. It provides an easy way to create unique and imaginative characters for storytelling, chatting and other purposes.

## Prerequisites
Before running the script, make sure you have Python3 and dependencies installed:

CPU:
```py
pip install -r requirements.txt
```

CUDA:
```py
pip install -r requirements-cuda.txt
```
When you run the script for the first time, the script will automatically download the required LLM and Stable Diffusion models

## Generation options
```--name``` This flag allows you to specify the character's name. If provided, the script will use the name you specify. If not provided, the script will use the Language Model (LLM) to generate a name for the character.

```--gender``` Use this parameter to specify the character's gender. If provided, the script will use the specified gender. Otherwise, LLM will choose the gender.

```--summary``` Use this flag to specify the character's summary. If you provide a summary, it will be used for the character. If not provided, the script will use LLM to generate a summary for the character.

```--personality``` This flag lets you specify the character's personality. If you provide a personality description, it will be used. If not provided, the script will use LLM to generate a personality description for the character.

```--greeting-message``` Use this flag to specify the character's greeting message for interacting with users. If provided, the script will use the specified greeting message. If not provided, LLM will generate a greeting message for the character.

```--avatar-prompt``` This flag allows you to specify the prompt for generating the character's avatar. If provided, the script will use the specified prompt for avatar generation. If not provided, the script will use LLM to generate the prompt for the avatar.

```--topic``` Specify the topic for character generation using this flag. Topics can include "Fantasy", "Anime", "Noir style detective", "Old mage master of lightning", or any other topic relevant to your character. The topic can influence the character's details and characteristics.

```--negative-prompt``` This flag is used to provide a negative prompt for Stable Diffusion. A negative prompt can be used to guide the generation of character avatars by specifying elements that should not be included in the avatar.

```--scenario``` Use this flag to specify the character's scenario. If you provide a scenario, it will be used for the character. If not provided, the script will use LLM to generate a scenario for the character.

```--example-messages``` Specify example messages for the character using this flag. If you provide example messages, they will be used for the character. If not provided, the script will use LLM to generate example messages for the character.

## Colab usage
1. Open the notebook in Google Colab by clicking the badge:
<a target="_blank" href="https://colab.research.google.com/drive/1JqkrtFXKalcmuMvST2VltoS1UVwoQINH">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

2. After opening the link, you will see the notebook within the Google Colab environment.
3. Make sure to check whether a GPU is selected for your environment. Running your script on a CPU will not work. To verify the GPU selection, follow these steps:
   1. Click on "Runtime" in the top menu.
   2. Change the CPU to one of these: T4 GPU, A100 GPU, V100 GPU
   3. Click "Save."
4. After the environment starts, you need to run each cell in turn
5. If everything is prepared, you can just run the last cell to generate characters

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
```
python ./app/main.py --name "Albert Einstein" --topic "science" --avatar-prompt "Albert Einstein"
```

## License
2023 Hubert Kasperek

This script is available under the AGPL-3.0 license. Details of the license can be found in the [LICENSE](LICENSE) file.
