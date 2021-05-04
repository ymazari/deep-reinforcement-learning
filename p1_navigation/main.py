from unityagents import UnityEnvironment
from p1_agent import DoubleDqnAgent
from utils import watch, interact, plot
from workspace_utils import active_session
import torch

if __name__ == "__main__":
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    enable_gpu = False  # Up to the user
    gpu_available = torch.cuda.is_available()  # Checks the environment
    train_on_gpu = enable_gpu and gpu_available
    if train_on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # watch an untrained agent
    untrained_agent = DoubleDqnAgent(state_size=state_size, action_size=action_size, seed=0, device=device)
    watch(untrained_agent, env, brain_name)

    agent = DoubleDqnAgent(state_size=state_size, action_size=action_size, seed=0, device=device)
    scores, best_avg_reward = interact(agent, env, brain_name, min_score=100.0)

    plot(scores)

    # watch the trained agent
    agent.qnetwork_local.load_state_dict(torch.load('top_model.pth'))
    watch(agent, env, brain_name)

    env.close()