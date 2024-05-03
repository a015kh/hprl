# Using custom env
# Karel custom env: cleanHouse, harvester, fourCorners, randomMaze, stairClimber, topOff, doorkey, oneStroke, seeder, snake
# class ExecEnv_option
# class ProgramEnv_option (task_definition == custom_reward)
# PPO for topOff_sparse (64 dim)
seed=0
# CUDA_VISIBLE_DEVICES="0" python3 pretrain/trainer_option_new_vae_L30.py --configfile pretrain/cfg_option_new_vae.py --datadir placeholder --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed  --num_lstm_cell_units 64 --net.saved_params_path pretrain/output_dir_new_vae_L40_1m_30epoch_20230104/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear-handwritten-123-20230110-110800/best_valid_params.ptp --net.num_rnn_encoder_units 256 --net.num_rnn_decoder_units 256 --net.use_linear True --net.latent_mean_pooling False --rl.envs.executable.task_definition custom_reward  --max_program_len 40 --dsl.max_program_len 40 --prefix PPO_option_topOff_sparse_L38_step5_dim64u256_recurrent_fixedInput --PPO.num_processes 16 --PPO.lr 1e-5 --PPO.num_steps 800 --PPO.num_mini_batch 10 --PPO.entropy_coef 0.05 --PPO.hidden_size 64 --PPO.decoder_deterministic True --log_interval 1 --save_interval 50 --log_video_interval 200 --max_episode_steps 5 --input_channel 8 --logging.wandb False --PPO.recurrent_policy True --fixed_input True \
#     --PPO.num_env_steps 5e4 \
#     --algorithm PPO_option \
#     --outdir pretrain/debug \
#     --seed $seed \
#     --input_height 12 \
#     --input_width 12 \
#     --num_envs 32 \
#     --env_task topOff

CUDA_VISIBLE_DEVICES="0" python3 pretrain/trainer_option_new_vae_L30.py --configfile pretrain/cfg_option_new_vae.py --datadir placeholder --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed  --num_lstm_cell_units 64 --net.saved_params_path pretrain/output_dir_new_vae_L40_1m_30epoch_20230104/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear-handwritten-123-20230110-110800/best_valid_params.ptp --net.num_rnn_encoder_units 256 --net.num_rnn_decoder_units 256 --net.use_linear True --net.latent_mean_pooling False --rl.envs.executable.task_definition custom_reward  --max_program_len 40 --dsl.max_program_len 40 --prefix PPO_option_topOff_sparse_L38_step5_dim64u256_recurrent_fixedInput --PPO.num_processes 16 --PPO.lr 1e-5 --PPO.num_steps 800 --PPO.num_mini_batch 10 --PPO.entropy_coef 0.05 --PPO.hidden_size 64 --PPO.decoder_deterministic True --log_interval 1 --save_interval 50 --log_video_interval 200 --max_episode_steps 5 --input_channel 8 --logging.wandb False --PPO.recurrent_policy True --fixed_input True \
    --PPO.num_env_steps 5e4 \
    --algorithm PPO_option_test \
    --outdir pretrain/debug \
    --seed $seed \
    --input_height 12 \
    --input_width 12 \
    --num_envs 32 \
    --env_task topOff
