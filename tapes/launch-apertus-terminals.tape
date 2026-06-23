Output docs/assets/launch-apertus-terminals.gif
Output docs/assets/launch-apertus-terminals.mp4

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 700
Set Theme "Dracula"
Set TypingSpeed 60ms
Set PlaybackSpeed 1.0

# Clean prompt for the recording. The launcher (slurm), account, partition and
# CSCS API key all come from the saved `sml init` config (~/.sml/config.yml), and
# the script passes --partition itself, so nothing else needs pinning here.
# NOTE: with the slurm launcher the node-terminal chord runs `srun --pty` directly
# on the compute node — no SSH hop, so SML_CLUSTER_SSH_HOST is not needed (that's
# only for the FirecREST launcher). Run this tape FROM the cluster head node.
Hide
Type `export PS1="$ "`
Enter
Type "clear"
Enter
Show

# Launch the 3-replica Apertus-1.5 experiment. The script pins everything
# (--replicas 3, --tui, vllm, reservation, framework-args), so this is one command.
Type "bash examples/clariden/cli/swiss-ai/1p5-experiments/Apertus-1.5-8B-SFT-RL-DPO-SDPO-Mix-Less-Refuse-Feedback.sh"
Enter

# Wait for SLURM to allocate and the per-replica health checks to report. The
# node-terminal chord ("t") is only offered once a replica advertises a node host,
# so don't shorten this past the time the Node column fills in. Eyeball + bump if
# the cluster is slow to schedule.
Sleep 240s

# --- Node-terminal chord -----------------------------------------------------
# Press "t" to open the prompt, then type the node number. With 3 replicas the
# node numbers are 0–2 (node_rank = index * nodes_per_replica; TP=4 on 4-GPU
# nodes => 1 node/replica). A single digit auto-opens at once (fewer than 10
# nodes), so no Enter is needed. The TUI suspends, drops into a shell on the
# node, and resumes when that shell exits.

# Node 0
Type "t"
Sleep 1s
Type "0"
Sleep 6s
Type "nvidia-smi"
Enter
Sleep 5s
Type "exit"
Enter
Sleep 3s

# Node 1
Type "t"
Sleep 1s
Type "1"
Sleep 6s
Type "nvidia-smi"
Enter
Sleep 5s
Type "exit"
Enter
Sleep 3s

# Node 2
Type "t"
Sleep 1s
Type "2"
Sleep 6s
Type "nvidia-smi"
Enter
Sleep 5s
Type "exit"
Enter
Sleep 3s

# Back in the live display: wait for the model to report HEALTHY. Health polling
# needs SML_CSCS_API_KEY (or a saved `sml init` config); without it the TUI only
# returns when the job terminates. The Makefile `demo-terminals` target speeds up
# everything from the TUI onward, so this long tail is fine.
Sleep 600s
