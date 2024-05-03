

python3 karel_env/calculate_reward_demo2_program.py \
    -o evaluation/predict_hprl \
    -c pretrain/cfg_option_new_vae.py \
    --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed \
    --rl.envs.executable.task_definition custom_reward \
    --algorithm CEM \
    --CEM.population_size 10 \
    --input_channel 8 \
    --input_height 8 \
    --input_width 8 \
    --max_program_len 200 \
    --dsl.max_program_len 200 \
    --num_demo 32 \
    --max_episode_steps 1 \
    --program_dir tasks/hc8

python3 karel_env/calculate_reward_demo2_program.py \
    -o evaluation/predict_hprl \
    -c pretrain/cfg_option_new_vae.py \
    --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed \
    --rl.envs.executable.task_definition custom_reward \
    --algorithm CEM \
    --CEM.population_size 10 \
    --input_channel 8 \
    --input_height 12 \
    --input_width 12 \
    --max_program_len 200 \
    --dsl.max_program_len 200 \
    --num_demo 32 \
    --max_episode_steps 1 \
    --program_dir tasks/hc12

python3 karel_env/calculate_reward_demo2_program.py \
    -o evaluation/predict_hprl \
    -c pretrain/cfg_option_new_vae.py \
    --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed \
    --rl.envs.executable.task_definition custom_reward \
    --algorithm CEM \
    --CEM.population_size 10 \
    --input_channel 8 \
    --input_height 14 \
    --input_width 22 \
    --max_program_len 200 \
    --dsl.max_program_len 200 \
    --num_demo 32 \
    --max_episode_steps 1 \
    --program_dir tasks/hc
