Output docs/assets/launch-apertus.gif
Output docs/assets/launch-apertus.mp4

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 700
Set Theme "Dracula"
Set TypingSpeed 60ms
Set PlaybackSpeed 1.0

# Pin infra choices via env vars so the demo focuses on model selection.
# Anything not pinned will be prompted for interactively.
Hide
Type `export PS1="$ "`
Enter
Type "export SML_FIRECREST_SYSTEM=clariden"
Enter
Type "export SML_PARTITION=normal"
Enter
Type "clear"
Enter
Show

Type "sml"
Enter
Sleep 4s

# Model picker — arrow down to swiss-ai/Apertus-8B-Instruct-2509 (sglang).
# Position 9 in src/swiss_ai_model_launch/assets/models.json (0-indexed).
Down@200ms 9
Sleep 800ms
Enter
Sleep 2000ms

# Framework picker. sglang
Enter
Sleep 3s

# Workers - use default
Enter
Sleep 3s

# Time limit — clear any default/leftover, then type explicitly.
Ctrl+a
Ctrl+k
Type "02:00:00"
Enter
Sleep 3s

# TUI / log viewer — recorded at 1x; the Makefile `demo` target post-processes
# everything from this point onward at 10x via ffmpeg.
Sleep 600s
