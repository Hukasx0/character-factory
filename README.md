# Character factory

WebUI using Mistral 7b instruct 0.1:
<a target="_blank" href="https://colab.research.google.com/drive/108koWoCDGaLZhZ0eV-gFuWtsnnLFMeCB">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

WebUI using Zephyr 7B beta:
<a target="_blank" href="https://colab.research.google.com/drive/1JqkrtFXKalcmuMvST2VltoS1UVwoQINH">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This Python script is designed to help you generate characters for [SillyTavern](https://github.com/SillyTavern/SillyTavern), [TavernAI](https://github.com/TavernAI/TavernAI), [TextGenerationWebUI](https://github.com/oobabooga/text-generation-webui) and many more, using LLM (Large Language Model) and Stable Diffusion. The script utilizes various deep learning models to create detailed character cards, including names, summaries, personalities, greeting messages, and character avatars.

<div>
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Grumpy_Purrsnatch/Grumpy_Purrsnatch.card.png?raw=true" width="300" height="300" alt="Grumpy Purrsnatch character card">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Lily_Harper/Lily_Harper.card.png?raw=true" width="300" height="300" alt="Lily Harper character card">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Arthondt_Lightbringer/Arthondt_Lightbringer.card.png?raw=true" width="300" height="300" alt="Arthondt Lightbringer character card">
  <img src="https://github.com/Hukasx0/character-factory/blob/main/examples/Albert_Einstein/Albert_Einstein.card.png?raw=true" width="300" height="300" alt="Albert Einstein character card">
</div>
(these four images above are valid character cards (V1), you can download them and use them in any frontend that supports character cards)

---

This script is designed to streamline the process of character generation for SillyTavern, TavernAI, and TextGenerationWebUI by leveraging LLM and Stable Diffusion models. It provides an easy way to create unique and imaginative characters for storytelling, chatting and other purposes.

## Running the script locally
### CPU
1. download miniconda from https://docs.conda.io/projects/miniconda/en/latest/
2. familiarize yourself with how conda works https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
3. Download Git (if you don't have it already) https://git-scm.com/
4. Clone git repository
```
git clone https://github.com/Hukasx0/character-factory
```
5. Open the anaconda prompt and enter the path of the folder

for example:
```
cd C:\Users\me\Desktop\character-factory
```
6. Execute these commands in the conda command prompt step by step.
```
conda create -n character-factory
```
```
conda activate character-factory
```
```
conda install python=3.11
```
```
pip install -r requirements.txt
```

and you can start using the script, for example like this:
```
python ./app/main-mistral.py --name "Albert Einstein" --topic "science" --avatar-prompt "Albert Einstein"
```

***Later, the next time you run it, you don't need to create a new environment, just repeat step 5. and type in (in the conda command prompt)***
```
conda activate character-factory
```

### CUDA
1. download miniconda from https://docs.conda.io/projects/miniconda/en/latest/
2. familiarize yourself with how conda works https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
3. Download Git (if you don't have it already) https://git-scm.com/
4. Clone git repository
```
git clone https://github.com/Hukasx0/character-factory
```
5. Open the anaconda prompt and enter the path of the folder

for example:
```
cd C:\Users\me\Desktop\character-factory
```
6. Download the Cuda package for Anaconda https://anaconda.org/nvidia/cuda
8. Execute these commands in the conda command prompt step by step.
```
conda create -n character-factory
```
```
conda activate character-factory
```
```
conda install python=3.11
```
```
pip install -r requirements-cuda.txt
```

and you can start using the script, for example like this:
```
python ./app/main-mistral.py --name "Albert Einstein" --topic "science" --avatar-prompt "Albert Einstein"
```

***Later, the next time you run it, you don't need to create a new environment, just repeat step 5. and type in (in the conda command prompt)***
```
conda activate character-factory
```
  
#### When you run the script for the first time, the script will automatically download the required LLM and Stable Diffusion models

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
1. Open the notebook in Google Colab by clicking one of those badges:

version using Mistral 7b instruct 0.1:
<a target="_blank" href="https://colab.research.google.com/drive/108koWoCDGaLZhZ0eV-gFuWtsnnLFMeCB">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

version using Zephyr 7B beta:
<a target="_blank" href="https://colab.research.google.com/drive/1JqkrtFXKalcmuMvST2VltoS1UVwoQINH">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

3. After opening the link, you will see the notebook within the Google Colab environment.
4. Make sure to check whether a GPU is selected for your environment. Running your script on a CPU will not work. To verify the GPU selection, follow these steps:
   1. Click on "Runtime" in the top menu.
   2. Change the CPU to one of these: T4 GPU, A100 GPU, V100 GPU
   3. Click "Save."
5. After the environment starts, you need to run each cell in turn
6. If everything is prepared, you can just run the last cell to generate characters

## Example usage:
```
python ./app/main-zephyr.py --topic "{{user}}'s pessimistic, monday-hating cat" --negative-prompt "human, gore, nsfw"
```
```
python ./app/main-zephyr.py --topic "{{user}}'s childhood friend, who secretly loves him" --gender "female" --negative-prompt "gore, nude, nsfw"
```
```
python ./app/main-mistral.py --topic "Old mage master of lightning" --gender "male" --negative-prompt "anime, nature, city, modern, young"
```
```
python ./app/main-mistral.py --name "Albert Einstein" --topic "science" --avatar-prompt "Albert Einstein"
```

## License
2023 Hubert Kasperek

This script is available under the AGPL-3.0 license. Details of the license can be found in the [LICENSE](LICENSE) file.
