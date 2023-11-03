#!/bin/bash

read -p "Enter character name (press Enter to skip): " name
read -p "Enter character summary (press Enter to skip): " summary
read -p "Enter character personality (press Enter to skip): " personality
read -p "Enter character scenario (press Enter to skip): " scenario
read -p "Enter greeting message (press Enter to skip): " greeting_message
read -p "Enter example messages (press Enter to skip): " example_messages
read -p "Enter topic (e.g., Fantasy, Anime, Warrior, Dwarf): " topic
read -p "Enter avatar prompt (press Enter to skip): " avatar_prompt
read -p "Enter negative prompt for Stable Diffusion (press Enter to skip): " negative_prompt

cmd="python ./app/main.py"

[[ ! -z "$name" ]] && cmd+=" --name \"$name\""
[[ ! -z "$summary" ]] && cmd+=" --summary \"$summary\""
[[ ! -z "$personality" ]] && cmd+=" --personality \"$personality\""
[[ ! -z "$scenario" ]] && cmd+=" --scenario \"$scenario\""
[[ ! -z "$greeting_message" ]] && cmd+=" --greeting-message \"$greeting_message\""
[[ ! -z "$example_messages" ]] && cmd+=" --example_messages \"$example_messages\""
[[ ! -z "$topic" ]] && cmd+=" --topic \"$topic\""
[[ ! -z "$avatar_prompt" ]] && cmd+=" --avatar-prompt \"$avatar_prompt\""
[[ ! -z "$negative_prompt" ]] && cmd+=" --negative-prompt \"$negative_prompt\""

eval $cmd

if [ $? -ne 0 ]; then
    read -p "Press enter to continue..."
fi
