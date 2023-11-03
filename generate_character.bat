@echo off
set /p name="Enter character name (press Enter to skip): "
set /p summary="Enter character summary (press Enter to skip): "
set /p personality="Enter character personality (press Enter to skip): "
set /p scenario="Enter character scenario (press Enter to skip): "
set /p greeting_message="Enter greeting message (press Enter to skip): "
set /p example_messages="Enter example messages (press Enter to skip): "
set /p avatar_prompt="Enter avatar prompt (press Enter to skip): "
set /p topic="Enter topic (e.g., Fantasy, Anime, Warrior, Dwarf): "
set /p negative_prompt="Enter negative prompt for Stable Diffusion (press Enter to skip): "

python ./app/main.py --name "%name%" --summary "%summary%" --personality "%personality%" --scenario "%scenario%" --greeting-message "%greeting_message%" --example_messages "%example_messages%" --avatar-prompt "%avatar_prompt%" --topic "%topic%" --negative-prompt "%negative_prompt%"
if %ERRORLEVEL% neq 0 pause
