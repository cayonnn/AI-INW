from src.rl.multi_agent_env import TradingEnvMultiAgent

env = TradingEnvMultiAgent()
obs = env.reset()

print("Multi-Agent Environment Test")
print("=" * 40)

for i in range(10):
    actions = {
        "alpha": {"entry": 1, "pyramid": 0, "aggression": 1.0},
        "guardian": {"risk_mult": 0.8, "freeze": False}
    }
    obs, rewards, done, info = env.step(actions)
    print(f"Step {i}: DD={info['dd_today']:.1f}%, Guard={info['guard_level']}, Score={info['score']:.1f}")
    if done:
        print("Episode ended!")
        break

print("=" * 40)
print("Test completed!")
