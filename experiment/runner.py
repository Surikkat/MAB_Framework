import json
import os
import time

class ExperimentRunner:
    def __init__(self, env, algorithm, steps, output_file="results.json"):
        self.env = env
        self.algorithm = algorithm
        self.steps = steps
        self.output_file = output_file
        self.results = {
            "rewards": [],
            "cumulative_reward": [],
            "regrets":[],
            "cumulative_regret": [],
            "average_regret":[],
            "times":[]
        }

    def run(self):
        
        cum_reward = 0.0
        cum_regret = 0.0

        for step in range(self.steps):
            start_time = time.time()
            context = self.env.get_context()
            action = self.algorithm.select_arm(context)
            step_result = self.env.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 2:
                reward, optimal_reward = step_result
            else:
                reward = step_result
                optimal_reward = reward
            
            self.algorithm.update(context, action, reward)

            iter_time = time.time() - start_time
            regret = optimal_reward - reward
            
            cum_reward += reward
            cum_regret += regret
            avg_regret = cum_regret / (step + 1)

            self.results["rewards"].append(reward)
            self.results["cumulative_reward"].append(cum_reward)
            self.results["regrets"].append(regret)
            self.results["cumulative_regret"].append(cum_regret)
            self.results["average_regret"].append(avg_regret)
            self.results["times"].append(iter_time)

        self.save_results()

    def save_results(self):
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)