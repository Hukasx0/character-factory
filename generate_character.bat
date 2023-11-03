@echo off
setlocal enabledelayedexpansion

set "args="

set /p "name=Enter character name (press Enter to skip): "
if not "!name!"=="" set "args=!args! --name "!name!""

set /p "summary=Enter character summary (press Enter to skip): "
if not "!summary!"=="" set "args=!args! --summary "!summary!""

set /p "personality=Enter character personality (press Enter to skip): "
if not "!personality!"=="" set "args=!args! --personality "!personality!""

set /p "scenario=Enter character scenario (press Enter to skip): "
if not "!scenario!"=="" set "args=!args! --scenario "!scenario!""

set /p "greeting_message=Enter greeting message (press Enter to skip): "
if not "!greeting_message!"=="" set "args=!args! --greeting-message "!greeting_message!""

set /p "example_messages=Enter example messages (press Enter to skip): "
if not "!example_messages!"=="" set "args=!args! --example_messages "!example_messages!""

set /p "topic=Enter topic (e.g., Fantasy, Anime, Warrior, Dwarf): "
if not "!topic!"=="" set "args=!args! --topic "!topic!""

set /p "avatar_prompt=Enter avatar prompt (press Enter to skip): "
if not "!avatar_prompt!"=="" set "args=!args! --avatar-prompt "!avatar_prompt!""

set /p "negative_prompt=Enter negative prompt for Stable Diffusion (press Enter to skip): "
if not "!negative_prompt!"=="" set "args=!args! --negative-prompt "!negative_prompt!""

echo Running command: python ./app/main.py !args!
python ./app/main.py !args!
if %ERRORLEVEL% neq 0 pause
