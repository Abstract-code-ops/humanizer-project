# Humanizer Skill (placeholder)

This is a placeholder for the vendored `blader/humanizer` skill referenced in
docs/decision-log.md (v2.11.2, MIT license, 35 "signs of AI writing" patterns
+ rewrite process). Replace this file with the actual vendored skill content
if you have redistribution rights to it.

`humanizers/skill_humanizer.py` reads this file at runtime and falls back to
a short built-in instruction if it's missing, so the skill_gemini / skill_deepseek
tiers still run without it — just with a weaker prompt.
